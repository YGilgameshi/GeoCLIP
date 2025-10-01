"""
GeoCLIP - 深度估计模块
基于预训练模型进行单目深度估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Tuple, Union, List
import timm
import warnings

from torchvision import transforms
warnings.filterwarnings("ignore")


class DepthEstimator(nn.Module):
    """
    单目深度估计器
    支持多种预训练模型: MiDaS, DPT
    """

    def __init__(self,
                 model_type: str = "DPT_Large",
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (384, 384)):
        super(DepthEstimator, self).__init__()

        self.model_type = model_type
        self.device = device
        self.input_size = input_size

        # 加载预训练深度估计模型
        self.depth_model = self._load_depth_model()
        self.depth_model.eval()

        # 预处理transforms
        self.transform = self._get_transform()

    def _load_depth_model(self):
        """加载预训练深度估计模型，包含错误处理和回退机制"""
        # 可用模型映射表
        model_mapping = {
            "dpt_hybrid_384": "DPT_Hybrid",  # 兼容旧名称
            "dpt_large_384": "DPT_Large",    # 兼容旧名称
            "midas_v21": "MiDaS",            # 兼容旧名称
            "midas_small": "MiDaS_small"     # 兼容旧名称
        }

        # 如果使用旧名称，映射到新名称
        if self.model_type in model_mapping:
            self.model_type = model_mapping[self.model_type]
            print(f"模型名称已更新为: {self.model_type}")

        try:
            # 尝试加载指定模型
            model = torch.hub.load('intel-isl/MiDaS', self.model_type, pretrained=True)
            print(f"成功加载模型: {self.model_type}")
        except RuntimeError as e:
            print(f"模型 {self.model_type} 加载失败: {e}")

            # 按优先级尝试回退模型
            fallback_models = ['DPT_Large', 'DPT_Hybrid', 'MiDaS', 'MiDaS_small']

            for fallback_model in fallback_models:
                if fallback_model != self.model_type:
                    try:
                        print(f"尝试回退模型: {fallback_model}")
                        model = torch.hub.load('intel-isl/MiDaS', fallback_model, pretrained=True)
                        self.model_type = fallback_model
                        print(f"成功加载回退模型: {fallback_model}")
                        break
                    except RuntimeError:
                        continue
            else:
                raise RuntimeError("所有模型都无法加载，请检查网络连接或MiDaS版本")

        return model.to(self.device)

    def _get_transform(self):
        """获取模型对应的预处理变换"""
        try:
            # 从MiDaS加载官方transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

            if self.model_type in ['DPT_Large', 'DPT_Hybrid']:
                return midas_transforms.dpt_transform
            else:
                return midas_transforms.small_transform

        except Exception as e:
            print(f"加载MiDaS transforms失败: {e}")
            print("使用自定义transforms")

            # 回退到自定义transforms
            from torchvision import transforms
            return transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def _preprocess_image(self, image) -> torch.Tensor:
        """预处理输入图像"""

        # 第一步：先处理类型转换，确保是tensor
        if not isinstance(image, torch.Tensor):
            if hasattr(image, 'convert'):  # PIL Image
                from torchvision.transforms import ToTensor
                image = ToTensor()(image.convert('RGB'))
            elif isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                raise TypeError(f"不支持的图像类型: {type(image)}")

        # 第二步：现在安全地处理维度
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # 第三步：确保在正确设备上
        image = image.to(self.device)

        # 第四步：应用变换
        if hasattr(self.transform, '__call__'):
            if image.dim() == 4:
                processed_images = []
                for i in range(image.size(0)):
                    img = image[i]

                    # 转换为numpy数组传递给MiDaS transform
                    img_numpy = img.permute(1, 2, 0).cpu().numpy()
                    if img_numpy.max() <= 1.0:
                        img_numpy = (img_numpy * 255).astype(np.uint8)
                    else:
                        img_numpy = img_numpy.astype(np.uint8)

                    # 调用MiDaS transform
                    processed_img = self.transform(img_numpy)

                    # 确保返回的是正确格式的tensor
                    if isinstance(processed_img, dict):
                        # MiDaS transform可能返回字典
                        processed_img = processed_img.get('image', processed_img)

                    if not isinstance(processed_img, torch.Tensor):
                        processed_img = torch.from_numpy(processed_img).float()

                    # 确保是3维 [C, H, W]
                    if processed_img.dim() == 2:
                        processed_img = processed_img.unsqueeze(0)
                    elif processed_img.dim() == 4:
                        processed_img = processed_img.squeeze(0)

                    processed_images.append(processed_img.to(self.device))

                result = torch.stack(processed_images)

                # 最终检查：确保是4维 [B, C, H, W]
                if result.dim() != 4:
                    raise ValueError(f"预处理后维度错误: {result.shape}, 期望4维 [B, C, H, W]")

                return result
            else:
                # 单个图像的情况
                img_numpy = image[0].permute(1, 2, 0).cpu().numpy()
                if img_numpy.max() <= 1.0:
                    img_numpy = (img_numpy * 255).astype(np.uint8)
                else:
                    img_numpy = img_numpy.astype(np.uint8)

                processed_img = self.transform(img_numpy)

                if isinstance(processed_img, dict):
                    processed_img = processed_img.get('image', processed_img)

                if not isinstance(processed_img, torch.Tensor):
                    processed_img = torch.from_numpy(processed_img).float()

                if processed_img.dim() == 2:
                    processed_img = processed_img.unsqueeze(0)
                elif processed_img.dim() == 4:
                    processed_img = processed_img.squeeze(0)

                if processed_img.dim() == 3:
                    processed_img = processed_img.unsqueeze(0)

                return processed_img.to(self.device)
        else:
            return image

    def estimate_depth(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        估计图像深度

        Args:
            image: 输入图像 [B, 3, H, W] 或 [3, H, W] 或 PIL Image 或 numpy array

        Returns:
            depth: 深度图 [B, 1, H, W]
        """
        with torch.no_grad():
            # 预处理图像
            processed_image = self._preprocess_image(image)
            original_shape = processed_image.shape[-2:]

            # 深度估计
            depth = self.depth_model(processed_image)

            # 后处理
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            elif depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(1)

            # 如果输出尺寸与输入不同，调整大小
            if depth.shape[-2:] != original_shape:
                depth = F.interpolate(depth, size=original_shape,
                                    mode='bilinear', align_corners=False)

            return depth

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.estimate_depth(image)

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'device': self.device,
            'input_size': self.input_size,
            'parameters': sum(p.numel() for p in self.depth_model.parameters())
        }


class MultiScaleDepthEstimator(nn.Module):
    """
    多尺度深度估计器
    在多个尺度上进行深度估计并融合结果
    """

    def __init__(self,
                 model_type: str = "DPT_Large",
                 scales: List[float] = [1.0, 0.75, 0.5],
                 device: str = "cuda"):
        super(MultiScaleDepthEstimator, self).__init__()

        self.scales = scales
        self.device = device

        # 创建单个深度估计器（所有尺度共享）
        self.depth_estimator = DepthEstimator(model_type, device)

        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        多尺度深度估计

        Args:
            image: 输入图像 [B, 3, H, W]

        Returns:
            depth: 融合后的深度图 [B, 1, H, W]
        """
        original_shape = image.shape[-2:]
        depth_maps = []

        for scale in self.scales:
            if scale != 1.0:
                # 缩放图像
                scaled_size = (int(original_shape[0] * scale),
                             int(original_shape[1] * scale))
                scaled_image = F.interpolate(image, size=scaled_size,
                                           mode='bilinear', align_corners=False)
            else:
                scaled_image = image

            # 估计深度
            depth = self.depth_estimator(scaled_image)

            # 缩放回原尺寸
            if depth.shape[-2:] != original_shape:
                depth = F.interpolate(depth, size=original_shape,
                                    mode='bilinear', align_corners=False)

            depth_maps.append(depth)

        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_depth = sum(w * d for w, d in zip(weights, depth_maps))

        return fused_depth


class DepthUncertaintyEstimator(nn.Module):
    """
    深度不确定性估计器
    同时估计深度值和不确定性
    """

    def __init__(self,
                 backbone: str = "efficientnet_b4",
                 pretrained: bool = True,
                 device: str = "cuda"):
        super(DepthUncertaintyEstimator, self).__init__()

        self.device = device

        # 使用timm加载backbone
        try:
            self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                            features_only=True)
        except Exception as e:
            print(f"加载backbone {backbone} 失败: {e}")
            print("使用默认backbone: resnet50")
            self.backbone = timm.create_model('resnet50', pretrained=pretrained,
                                            features_only=True)

        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dims = [f.shape[1] for f in features]

        # 深度预测头
        self.depth_head = self._build_decoder_head(feature_dims, 1)

        # 不确定性预测头
        self.uncertainty_head = self._build_decoder_head(feature_dims, 1)

    def _build_decoder_head(self, feature_dims: List[int], output_channels: int):
        """构建解码器头部"""
        layers = []

        # 从最深的特征开始
        in_channels = feature_dims[-1]

        for i in range(len(feature_dims) - 1, 0, -1):
            layers.extend([
                nn.Conv2d(in_channels, feature_dims[i - 1], 3, padding=1),
                nn.BatchNorm2d(feature_dims[i - 1]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ])
            in_channels = feature_dims[i - 1]

        # 最终输出层
        layers.extend([
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, 1)
        ])

        return nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            image: 输入图像 [B, 3, H, W]

        Returns:
            depth: 深度图 [B, 1, H, W]
            uncertainty: 不确定性图 [B, 1, H, W]
        """
        # 提取多尺度特征
        features = self.backbone(image)

        # 预测深度
        depth = self.depth_head(features[-1])
        depth = torch.relu(depth)  # 确保深度为正

        # 预测不确定性
        uncertainty = self.uncertainty_head(features[-1])
        uncertainty = torch.sigmoid(uncertainty)  # 归一化到[0,1]

        # Resize到输入尺寸
        target_size = image.shape[-2:]
        depth = F.interpolate(depth, size=target_size,
                            mode='bilinear', align_corners=False)
        uncertainty = F.interpolate(uncertainty, size=target_size,
                                  mode='bilinear', align_corners=False)

        return depth, uncertainty


def create_depth_estimator(config: dict) -> nn.Module:
    """
    工厂函数：根据配置创建深度估计器

    Args:
        config: 配置字典

    Returns:
        depth_estimator: 深度估计器实例
    """
    estimator_type = config.get('type', 'single')

    if estimator_type == 'single':
        return DepthEstimator(
            model_type=config.get('model_type', 'DPT_Large'),
            device=config.get('device', 'cuda'),
            input_size=config.get('input_size', (384, 384))
        )
    elif estimator_type == 'multiscale':
        return MultiScaleDepthEstimator(
            model_type=config.get('model_type', 'DPT_Large'),
            scales=config.get('scales', [1.0, 0.75, 0.5]),
            device=config.get('device', 'cuda')
        )
    elif estimator_type == 'uncertainty':
        return DepthUncertaintyEstimator(
            backbone=config.get('backbone', 'efficientnet_b4'),
            pretrained=config.get('pretrained', True),
            device=config.get('device', 'cuda')
        )
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


def test_depth_estimator():
    """测试深度估计器"""
    try:
        print("=== 测试深度估计器 ===")

        # 测试单尺度估计器
        print("\n1. 测试单尺度深度估计器")
        depth_estimator = DepthEstimator()
        print(f"模型信息: {depth_estimator.get_model_info()}")

        # 创建测试图像
        dummy_image = torch.randn(2, 3, 256, 256)
        if torch.cuda.is_available():
            dummy_image = dummy_image.cuda()

        # 估计深度
        depth = depth_estimator(dummy_image)
        print(f"输入图像形状: {dummy_image.shape}")
        print(f"输出深度图形状: {depth.shape}")
        print(f"深度范围: {depth.min():.3f} - {depth.max():.3f}")

        # 测试多尺度估计器
        print("\n2. 测试多尺度深度估计器")
        multiscale_estimator = MultiScaleDepthEstimator()
        depth_ms = multiscale_estimator(dummy_image)
        print(f"多尺度深度图形状: {depth_ms.shape}")
        print(f"多尺度深度范围: {depth_ms.min():.3f} - {depth_ms.max():.3f}")

        print("\n✅ 所有测试通过!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


# 示例使用
if __name__ == "__main__":
    test_depth_estimator()