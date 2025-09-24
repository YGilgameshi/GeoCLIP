

"""
GeoCLIP - 损失函数模块
包含对比学习、几何一致性、异常检测等损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math


class ContrastiveLoss(nn.Module):
    """
    对比学习损失 - 用于2D和3D特征对齐
    """

    def __init__(self,
                 temperature: float = 0.07,
                 normalize: bool = True):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self,
                features_2d: torch.Tensor,
                features_3d: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算对比学习损失

        Args:
            features_2d: 2D特征 [B, D]
            features_3d: 3D特征 [B, D]
            labels: 标签 [B] (可选，用于监督对比学习)

        Returns:
            loss: 对比损失
        """
        batch_size = features_2d.size(0)
        device = features_2d.device

        # 标准化特征
        if self.normalize:
            features_2d = F.normalize(features_2d, dim=1)
            features_3d = F.normalize(features_3d, dim=1)

        # 计算相似性矩阵
        similarity_matrix = torch.matmul(features_2d, features_3d.t()) / self.temperature

        # 创建正样本mask
        if labels is not None:
            # 监督对比学习：相同标签为正样本
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.t()).float().to(device)
        else:
            # 自监督对比学习：对角线为正样本
            mask = torch.eye(batch_size, device=device)

        # 计算对比损失
        # 分子：正样本的相似性
        pos_similarity = similarity_matrix * mask

        # 分母：所有样本的相似性（除了自己）
        neg_mask = 1 - mask
        exp_sim = torch.exp(similarity_matrix)

        # 计算损失
        pos_sum = torch.sum(exp_sim * mask, dim=1)
        neg_sum = torch.sum(exp_sim * neg_mask, dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))

        return loss.mean()


class GeometryConsistencyLoss(nn.Module):
    """
    几何一致性损失 - 确保2D和3D特征的几何关系一致
    """

    def __init__(self,
                 consistency_type: str = "distance",
                 lambda_weight: float = 1.0):
        super(GeometryConsistencyLoss, self).__init__()
        self.consistency_type = consistency_type
        self.lambda_weight = lambda_weight

    def forward(self,
                features_2d: torch.Tensor,
                features_3d: torch.Tensor,
                depth_maps: torch.Tensor) -> torch.Tensor:
        """
        计算几何一致性损失

        Args:
            features_2d: 2D特征 [B, D]
            features_3d: 3D特征 [B, D]
            depth_maps: 深度图 [B, 1, H, W]

        Returns:
            loss: 几何一致性损失
        """
        if self.consistency_type == "distance":
            return self._distance_consistency_loss(features_2d, features_3d, depth_maps)
        elif self.consistency_type == "ranking":
            return self._ranking_consistency_loss(features_2d, features_3d, depth_maps)
        else:
            raise ValueError(f"不支持的一致性类型: {self.consistency_type}")

    def _distance_consistency_loss(self, features_2d, features_3d, depth_maps):
        """基于距离的一致性损失"""
        batch_size = features_2d.size(0)

        # 计算特征距离矩阵
        dist_2d = torch.cdist(features_2d, features_2d)  # [B, B]
        dist_3d = torch.cdist(features_3d, features_3d)  # [B, B]

        # 计算深度统计量作为几何约束
        depth_stats = []
        for b in range(batch_size):
            depth = depth_maps[b, 0]  # [H, W]
            mean_depth = depth.mean()
            std_depth = depth.std()
            depth_stats.append(torch.stack([mean_depth, std_depth]))

        depth_features = torch.stack(depth_stats)  # [B, 2]
        dist_depth = torch.cdist(depth_features, depth_features)  # [B, B]

        # 一致性损失：特征距离应该与深度距离相关
        loss_2d = F.mse_loss(dist_2d, dist_depth)
        loss_3d = F.mse_loss(dist_3d, dist_depth)

        return self.lambda_weight * (loss_2d + loss_3d) / 2

    def _ranking_consistency_loss(self, features_2d, features_3d, depth_maps):
        """基于排序的一致性损失"""
        batch_size = features_2d.size(0)

        # 计算深度排序
        depth_means = torch.stack([depth_maps[b, 0].mean() for b in range(batch_size)])
        depth_order = torch.argsort(depth_means)

        # 计算特征相似性排序
        sim_2d = torch.matmul(F.normalize(features_2d, dim=1),
                              F.normalize(features_2d, dim=1).t())
        sim_3d = torch.matmul(F.normalize(features_3d, dim=1),
                              F.normalize(features_3d, dim=1).t())

        # 排序一致性损失
        loss = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # 深度顺序
                depth_order_ij = 1 if depth_means[i] < depth_means[j] else -1

                # 特征相似性顺序
                sim_2d_order = 1 if sim_2d[i, j] > sim_2d[j, i] else -1
                sim_3d_order = 1 if sim_3d[i, j] > sim_3d[j, i] else -1

                # 一致性损失
                loss += torch.relu(1 - depth_order_ij * sim_2d_order) ** 2
                loss += torch.relu(1 - depth_order_ij * sim_3d_order) ** 2

        return self.lambda_weight * loss / (batch_size * (batch_size - 1))


class AnomalyDetectionLoss(nn.Module):
    """
    异常检测损失函数
    """

    def __init__(self,
                 loss_type: str = "focal",
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "mean"):
        super(AnomalyDetectionLoss, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算异常检测损失

        Args:
            predictions: 预测结果 [B] 或 [B, num_classes]
            targets: 真实标签 [B]
            sample_weights: 样本权重 [B] (可选)

        Returns:
            loss: 异常检测损失
        """
        # 确保targets在有效范围内
        if predictions.dim() > 1:
            num_classes = predictions.shape[1]
            targets = torch.clamp(targets, 0, num_classes - 1)
        else:
            # 对于回归任务，确保targets在[0,1]范围内
            targets = torch.clamp(targets.float(), 0.0, 1.0)

        if self.loss_type == "focal":
            return self._focal_loss(predictions, targets, sample_weights)
        elif self.loss_type == "weighted_bce":
            return self._weighted_bce_loss(predictions, targets, sample_weights)
        elif self.loss_type == "dice":
            return self._dice_loss(predictions, targets, sample_weights)
        elif self.loss_type == "mse":
            return self._mse_loss(predictions, targets, sample_weights)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

    def _focal_loss(self, predictions, targets, sample_weights):
        """Focal Loss - 处理类别不平衡"""
        if predictions.dim() == 1:
            # 回归情况，转换为二分类概率
            predictions = predictions.unsqueeze(1)
            prob_neg = 1 - predictions
            probs = torch.cat([prob_neg, predictions], dim=1)
            targets = targets.long()
        else:
            # 分类情况
            probs = F.softmax(predictions, dim=1)
            targets = targets.long()

        # 确保targets在有效范围内
        num_classes = probs.shape[1]
        targets = torch.clamp(targets, 0, num_classes - 1)

        # 计算交叉熵
        ce_loss = F.cross_entropy(probs, targets, reduction='none')

        # 计算概率
        pt = torch.gather(probs, 1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # 加权损失
        focal_loss = focal_weight * ce_loss

        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def _weighted_bce_loss(self, predictions, targets, sample_weights):
        """加权二元交叉熵损失"""
        if predictions.dim() > 1:
            predictions = predictions[:, -1]  # 取最后一个类别的概率

        targets = targets.float()

        # 计算类别权重
        pos_count = (targets == 1).sum().float()
        neg_count = (targets == 0).sum().float()

        if pos_count > 0 and neg_count > 0:
            pos_weight = neg_count / pos_count
        else:
            pos_weight = 1.0

        # BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets,
            pos_weight=pos_weight,
            reduction='none'
        )

        if sample_weights is not None:
            bce_loss = bce_loss * sample_weights

        if self.reduction == "mean":
            return bce_loss.mean()
        elif self.reduction == "sum":
            return bce_loss.sum()
        else:
            return bce_loss

    def _dice_loss(self, predictions, targets, sample_weights):
        """Dice Loss - 处理分割任务"""
        if predictions.dim() > 1:
            predictions = F.softmax(predictions, dim=1)[:, -1]

        targets = targets.float()

        # Dice系数
        intersection = (predictions * targets).sum()
        dice_coeff = (2 * intersection + 1e-8) / (predictions.sum() + targets.sum() + 1e-8)

        # Dice损失
        dice_loss = 1 - dice_coeff

        if sample_weights is not None:
            dice_loss = dice_loss * sample_weights.mean()

        return dice_loss

    def _mse_loss(self, predictions, targets, sample_weights):
        """均方误差损失 - 用于回归任务"""
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)

        targets = targets.float()

        mse_loss = F.mse_loss(predictions, targets, reduction='none')

        if sample_weights is not None:
            mse_loss = mse_loss * sample_weights

        if self.reduction == "mean":
            return mse_loss.mean()
        elif self.reduction == "sum":
            return mse_loss.sum()
        else:
            return mse_loss


class GeoCLIPLoss(nn.Module):
    """
    GeoCLIP综合损失函数
    结合对比学习、几何一致性和异常检测损失
    """

    def __init__(self,
                 contrastive_weight: float = 1.0,
                 geometry_weight: float = 0.5,
                 anomaly_weight: float = 2.0,
                 use_contrastive: bool = True,
                 use_geometry: bool = True,
                 contrastive_config: Dict = None,
                 geometry_config: Dict = None,
                 anomaly_config: Dict = None):
        super(GeoCLIPLoss, self).__init__()

        self.contrastive_weight = contrastive_weight
        self.geometry_weight = geometry_weight
        self.anomaly_weight = anomaly_weight
        self.use_contrastive = use_contrastive
        self.use_geometry = use_geometry

        # 初始化各个损失函数
        if use_contrastive:
            contrastive_config = contrastive_config or {}
            self.contrastive_loss = ContrastiveLoss(**contrastive_config)

        if use_geometry:
            geometry_config = geometry_config or {}
            self.geometry_loss = GeometryConsistencyLoss(**geometry_config)

        anomaly_config = anomaly_config or {}
        self.anomaly_loss = AnomalyDetectionLoss(**anomaly_config)

    def forward(self,
                model_outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算GeoCLIP综合损失

        Args:
            model_outputs: 模型输出字典，包含：
                - 'anomaly_predictions': 异常预测
                - 'clip_features': CLIP特征
                - 'geometry_features': 几何特征
                - 'fused_features': 融合特征
                - 'depth_maps': 深度图
            targets: 目标字典，包含：
                - 'anomaly_labels': 异常标签
                - 'class_labels': 类别标签 (可选)

        Returns:
            loss_dict: 损失字典
        """
        loss_dict = {}
        total_loss = 0

        # 1. 异常检测损失
        anomaly_loss = self.anomaly_loss(
            model_outputs['anomaly_predictions'],
            targets['anomaly_labels']
        )
        loss_dict['anomaly_loss'] = anomaly_loss
        total_loss += self.anomaly_weight * anomaly_loss

        # 2. 对比学习损失
        if (self.use_contrastive and
                'clip_features' in model_outputs and
                'geometry_features' in model_outputs):

            clip_feat = model_outputs['clip_features']
            geometry_feat = model_outputs['geometry_features']

            # 确保特征维度匹配
            if clip_feat.shape[1] != geometry_feat.shape[1]:
                # 投影到相同维度
                min_dim = min(clip_feat.shape[1], geometry_feat.shape[1])
                if clip_feat.shape[1] > min_dim:
                    clip_feat = clip_feat[:, :min_dim]
                if geometry_feat.shape[1] > min_dim:
                    geometry_feat = geometry_feat[:, :min_dim]

            contrastive_loss = self.contrastive_loss(
                clip_feat,
                geometry_feat,
                targets.get('class_labels', None)
            )
            loss_dict['contrastive_loss'] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss

        # 3. 几何一致性损失
        if (self.use_geometry and
                'depth_maps' in model_outputs and
                'clip_features' in model_outputs and
                'geometry_features' in model_outputs):

            clip_feat = model_outputs['clip_features']
            geometry_feat = model_outputs['geometry_features']

            # 确保特征维度匹配
            if clip_feat.shape[1] != geometry_feat.shape[1]:
                min_dim = min(clip_feat.shape[1], geometry_feat.shape[1])
                if clip_feat.shape[1] > min_dim:
                    clip_feat = clip_feat[:, :min_dim]
                if geometry_feat.shape[1] > min_dim:
                    geometry_feat = geometry_feat[:, :min_dim]

            try:
                geometry_loss = self.geometry_loss(
                    clip_feat,
                    geometry_feat,
                    model_outputs['depth_maps']
                )
                loss_dict['geometry_loss'] = geometry_loss
                total_loss += self.geometry_weight * geometry_loss
            except Exception as e:
                print(f"几何一致性损失计算失败，跳过: {e}")
                loss_dict['geometry_loss'] = torch.tensor(0.0, device=clip_feat.device)

        loss_dict['total_loss'] = total_loss

        return loss_dict


class SelfSupervisedLoss(nn.Module):
    """
    自监督学习损失
    用于无标签数据的预训练
    """

    def __init__(self,
                 reconstruction_weight: float = 1.0,
                 depth_consistency_weight: float = 0.5,
                 feature_consistency_weight: float = 0.3):
        super(SelfSupervisedLoss, self).__init__()

        self.reconstruction_weight = reconstruction_weight
        self.depth_consistency_weight = depth_consistency_weight
        self.feature_consistency_weight = feature_consistency_weight

        # 重建损失
        self.reconstruction_loss = nn.MSELoss()

        # 特征一致性损失
        self.feature_consistency_loss = nn.CosineSimilarity(dim=1)

    def forward(self,
                original_images: torch.Tensor,
                reconstructed_images: torch.Tensor,
                original_depth: torch.Tensor,
                estimated_depth: torch.Tensor,
                features_aug1: torch.Tensor,
                features_aug2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算自监督损失

        Args:
            original_images: 原始图像
            reconstructed_images: 重建图像
            original_depth: 原始深度
            estimated_depth: 估计深度
            features_aug1: 增强版本1的特征
            features_aug2: 增强版本2的特征

        Returns:
            loss_dict: 损失字典
        """
        loss_dict = {}

        # 1. 图像重建损失
        recon_loss = self.reconstruction_loss(reconstructed_images, original_images)
        loss_dict['reconstruction_loss'] = recon_loss

        # 2. 深度一致性损失
        depth_loss = self.reconstruction_loss(estimated_depth, original_depth)
        loss_dict['depth_consistency_loss'] = depth_loss

        # 3. 特征一致性损失（数据增强不变性）
        feature_sim = self.feature_consistency_loss(features_aug1, features_aug2)
        feature_loss = 1 - feature_sim.mean()  # 最大化相似性
        loss_dict['feature_consistency_loss'] = feature_loss

        # 总损失
        total_loss = (self.reconstruction_weight * recon_loss +
                      self.depth_consistency_weight * depth_loss +
                      self.feature_consistency_weight * feature_loss)

        loss_dict['total_loss'] = total_loss

        return loss_dict


class AdversarialLoss(nn.Module):
    """
    对抗损失函数
    用于生成对抗训练或域适应
    """

    def __init__(self,
                 loss_type: str = "wgan",
                 lambda_gp: float = 10.0):
        super(AdversarialLoss, self).__init__()

        self.loss_type = loss_type
        self.lambda_gp = lambda_gp

    def forward(self,
                discriminator_real: torch.Tensor,
                discriminator_fake: torch.Tensor,
                real_data: Optional[torch.Tensor] = None,
                fake_data: Optional[torch.Tensor] = None,
                discriminator: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """
        计算对抗损失

        Args:
            discriminator_real: 判别器对真实数据的输出
            discriminator_fake: 判别器对生成数据的输出
            real_data: 真实数据 (用于梯度惩罚)
            fake_data: 生成数据 (用于梯度惩罚)
            discriminator: 判别器模型 (用于梯度惩罚)

        Returns:
            loss_dict: 损失字典
        """
        loss_dict = {}

        if self.loss_type == "wgan":
            # Wasserstein GAN损失
            d_loss = discriminator_real.mean() - discriminator_fake.mean()
            g_loss = -discriminator_fake.mean()

            # 梯度惩罚
            if real_data is not None and fake_data is not None and discriminator is not None:
                gp_loss = self._gradient_penalty(discriminator, real_data, fake_data)
                d_loss += self.lambda_gp * gp_loss
                loss_dict['gradient_penalty'] = gp_loss

        elif self.loss_type == "lsgan":
            # Least Squares GAN损失
            d_loss = 0.5 * (torch.mean((discriminator_real - 1) ** 2) +
                            torch.mean(discriminator_fake ** 2))
            g_loss = 0.5 * torch.mean((discriminator_fake - 1) ** 2)

        else:
            # 标准GAN损失
            d_loss = -(torch.log(discriminator_real + 1e-8).mean() +
                       torch.log(1 - discriminator_fake + 1e-8).mean())
            g_loss = -torch.log(discriminator_fake + 1e-8).mean()

        loss_dict['discriminator_loss'] = d_loss
        loss_dict['generator_loss'] = g_loss

        return loss_dict

    def _gradient_penalty(self, discriminator, real_data, fake_data):
        """计算梯度惩罚"""
        batch_size = real_data.size(0)
        device = real_data.device

        # 随机插值
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # 计算判别器输出
        d_interpolated = discriminator(interpolated)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建损失函数

    Args:
        config: 损失函数配置

    Returns:
        loss_function: 损失函数实例
    """
    loss_type = config.get('type', 'geoclip')

    if loss_type == 'geoclip':
        return GeoCLIPLoss(
            contrastive_weight=config.get('contrastive_weight', 1.0),
            geometry_weight=config.get('geometry_weight', 0.5),
            anomaly_weight=config.get('anomaly_weight', 2.0),
            use_contrastive=config.get('use_contrastive', True),
            use_geometry=config.get('use_geometry', True),
            contrastive_config=config.get('contrastive_config', {}),
            geometry_config=config.get('geometry_config', {}),
            anomaly_config=config.get('anomaly_config', {})
        )
    elif loss_type == 'self_supervised':
        return SelfSupervisedLoss(
            reconstruction_weight=config.get('reconstruction_weight', 1.0),
            depth_consistency_weight=config.get('depth_consistency_weight', 0.5),
            feature_consistency_weight=config.get('feature_consistency_weight', 0.3)
        )
    elif loss_type == 'adversarial':
        return AdversarialLoss(
            loss_type=config.get('adversarial_type', 'wgan'),
            lambda_gp=config.get('lambda_gp', 10.0)
        )
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")


# 测试函数
def test_loss_functions():
    """测试损失函数"""
    print("=== 测试GeoCLIP损失函数 ===")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 4
        feature_dim = 512

        # 1. 测试GeoCLIP综合损失
        print("\n1. 测试GeoCLIP综合损失")

        geoclip_loss = GeoCLIPLoss(
            contrastive_weight=1.0,
            geometry_weight=0.5,
            anomaly_weight=2.0
        )

        # 创建模拟数据
        model_outputs = {
            'anomaly_predictions': torch.randn(batch_size, 1, device=device).sigmoid(),
            'clip_features': torch.randn(batch_size, feature_dim, device=device),
            'geometry_features': torch.randn(batch_size, feature_dim, device=device),
            'depth_maps': torch.rand(batch_size, 1, 64, 64, device=device) * 5
        }

        targets = {
            'anomaly_labels': torch.randint(0, 2, (batch_size,), device=device),
            'class_labels': torch.randint(0, 5, (batch_size,), device=device)
        }

        loss_dict = geoclip_loss(model_outputs, targets)

        print(f"✅ GeoCLIP损失计算成功:")
        for key, value in loss_dict.items():
            print(f"   {key}: {value.item():.4f}")

        # 2. 测试对比学习损失
        print("\n2. 测试对比学习损失")
        contrastive_loss = ContrastiveLoss(temperature=0.07)

        features_2d = torch.randn(batch_size, feature_dim, device=device)
        features_3d = torch.randn(batch_size, feature_dim, device=device)

        contrast_loss = contrastive_loss(features_2d, features_3d)
        print(f"✅ 对比学习损失: {contrast_loss.item():.4f}")

        # 3. 测试几何一致性损失
        print("\n3. 测试几何一致性损失")
        geometry_loss = GeometryConsistencyLoss(consistency_type="distance")

        depth_maps = torch.rand(batch_size, 1, 32, 32, device=device) * 10
        geom_loss = geometry_loss(features_2d, features_3d, depth_maps)
        print(f"✅ 几何一致性损失: {geom_loss.item():.4f}")

        # 4. 测试异常检测损失
        print("\n4. 测试异常检测损失")
        anomaly_loss = AnomalyDetectionLoss(loss_type="focal")

        predictions = torch.randn(batch_size, device=device).sigmoid()
        labels = torch.randint(0, 2, (batch_size,), device=device)

        anom_loss = anomaly_loss(predictions, labels)
        print(f"✅ 异常检测损失: {anom_loss.item():.4f}")

        # 5. 测试自监督损失
        print("\n5. 测试自监督损失")
        ssl_loss = SelfSupervisedLoss()

        orig_imgs = torch.randn(batch_size, 3, 64, 64, device=device)
        recon_imgs = torch.randn(batch_size, 3, 64, 64, device=device)
        orig_depth = torch.rand(batch_size, 1, 64, 64, device=device)
        est_depth = torch.rand(batch_size, 1, 64, 64, device=device)
        feat_aug1 = torch.randn(batch_size, feature_dim, device=device)
        feat_aug2 = torch.randn(batch_size, feature_dim, device=device)

        ssl_loss_dict = ssl_loss(orig_imgs, recon_imgs, orig_depth,
                                 est_depth, feat_aug1, feat_aug2)

        print(f"✅ 自监督损失:")
        for key, value in ssl_loss_dict.items():
            print(f"   {key}: {value.item():.4f}")

        print("\n🎉 损失函数测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_loss_functions()