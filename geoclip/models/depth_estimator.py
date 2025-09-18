"""
GeoCLIP - 深度估计模块
基于预训练模型进行单目深度估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Optional, Tuple, Union
import timm


class DepthEstimator(nn.Module):
    """
    单目深度估计器
    支持多种预训练模型: MiDaS, DPT, LeReS
    """

    def __init__(self,
                 model_type: str = "dpt_hybrid_384",
                 device: str = "cuda",
                 input_size: Tuple[int, int] = (384, 384)):
        super(DepthEstimator, self).__init__()

        self.model_type = model_type
        self.device = device
        self.input_size = input_size

        # 加载预训练深度估计模型
        self.depth_model = self._load_depth_model()
        self.depth_model.eval()

        # 预处理和后处理的transforms
        self.preprocess = self._get_preprocess_transform()

    def _load_depth_model(self):
        """加载预训练深度估计模型"""
        if "dpt" in self.model_type:
            # 使用DPT模型
            model = torch.hub.load('intel-isl/MiDaS', self.model_type, pretrained=True)
        elif "midas" in self.model_type:
            # 使用MiDaS模型
            model = torch.hub.load('intel-isl/MiDaS', 'MiDaS', pretrained=True)
        else:
            raise ValueError(f"Unsupported depth model: {self.model_type}")

        return model.to(self.device)

    def _get_preprocess_transform(self):
        """获取预处理变换"""
        from torchvision import transforms

        if "dpt" in self.model_type:
            # DPT模型的预处理
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # MiDaS模型的预处理
            transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        return transform

    def estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        """
        估计图像深度

        Args:
            image: 输入图像 [B, 3, H, W]

        Returns:
            depth: 深度图 [B, 1, H, W]
        """
        with torch.no_grad():
            # 预处理
            if image.dim() == 3:
                image = image.unsqueeze(0)

            original_shape = image.shape[-2:]

            # Resize to model input size
            resized_image = F.interpolate(image, size=self.input_size,
                                          mode='bilinear', align_corners=False)

            # 应用预处理变换
            if hasattr(self, 'preprocess'):
                processed_image = self.preprocess(resized_image)
            else:
                processed_image = resized_image

            # 深度估计
            depth = self.depth_model(processed_image)

            # 后处理：确保深度值为正数
            depth = torch.relu(depth)

            # Resize back to original image size
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)
            elif depth.dim() == 2:
                depth = depth.unsqueeze(0).unsqueeze(1)

            depth = F.interpolate(depth, size=original_shape,
                                  mode='bilinear', align_corners=False)

            return depth

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.estimate_depth(image)


class MultiScaleDepthEstimator(nn.Module):
    """
    多尺度深度估计器
    在多个尺度上进行深度估计并融合结果
    """

    def __init__(self,
                 model_type: str = "dpt_hybrid_384",
                 scales: list = [1.0, 0.75, 0.5],
                 device: str = "cuda"):
        super(MultiScaleDepthEstimator, self).__init__()

        self.scales = scales
        self.device = device

        # 为每个尺度创建深度估计器
        self.depth_estimators = nn.ModuleList([
            DepthEstimator(model_type, device) for _ in scales
        ])

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

        for i, (scale, estimator) in enumerate(zip(self.scales, self.depth_estimators)):
            if scale != 1.0:
                # 缩放图像
                scaled_size = (int(original_shape[0] * scale),
                               int(original_shape[1] * scale))
                scaled_image = F.interpolate(image, size=scaled_size,
                                             mode='bilinear', align_corners=False)
            else:
                scaled_image = image

            # 估计深度
            depth = estimator(scaled_image)

            # 缩放回原尺寸
            if scale != 1.0:
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
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
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

    def _build_decoder_head(self, feature_dims: list, output_channels: int):
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
            model_type=config.get('model_type', 'dpt_hybrid_384'),
            device=config.get('device', 'cuda'),
            input_size=config.get('input_size', (384, 384))
        )
    elif estimator_type == 'multiscale':
        return MultiScaleDepthEstimator(
            model_type=config.get('model_type', 'dpt_hybrid_384'),
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


# 示例使用
if __name__ == "__main__":
    # 创建深度估计器
    depth_estimator = DepthEstimator()

    # 测试
    dummy_image = torch.randn(2, 3, 256, 256).cuda()
    depth = depth_estimator(dummy_image)

    print(f"输入图像形状: {dummy_image.shape}")
    print(f"输出深度图形状: {depth.shape}")
    print(f"深度范围: {depth.min():.3f} - {depth.max():.3f}")
    