"""
GeoCLIP - 主模型
整合2D CLIP特征、3D几何特征和深度信息进行异常检测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import open_clip

# 导入GeoCLIP组件
from geoclip.models.depth_estimator import DepthEstimator
from geoclip.models.geometry_encoder import create_geometry_encoder, VoxelEncoder
from geoclip.utils.voxel_utils import DepthToVoxelConverter


class FeatureFusionModule(nn.Module):
    """
    2D CLIP特征与3D几何特征融合模块
    """

    def __init__(self,
                 clip_dim: int = 512,
                 geometry_dim: int = 512,
                 fusion_dim: int = 1024,
                 output_dim: int = 512,
                 fusion_type: str = "cross_attention"):
        super(FeatureFusionModule, self).__init__()

        self.fusion_type = fusion_type
        self.clip_dim = clip_dim
        self.geometry_dim = geometry_dim
        self.fusion_dim = fusion_dim
        self.output_dim = output_dim

        if fusion_type == "cross_attention":
            self._build_cross_attention()
        elif fusion_type == "adaptive_fusion":
            self._build_adaptive_fusion()
        elif fusion_type == "simple_concat":
            self._build_simple_concat()
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")

    def _build_cross_attention(self):
        """构建交叉注意力融合"""
        # 投影到相同维度
        self.clip_proj = nn.Linear(self.clip_dim, self.fusion_dim)
        self.geometry_proj = nn.Linear(self.geometry_dim, self.fusion_dim)

        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def _build_adaptive_fusion(self):
        """构建自适应融合"""
        # 特征投影
        self.clip_proj = nn.Linear(self.clip_dim, self.fusion_dim)
        self.geometry_proj = nn.Linear(self.geometry_dim, self.fusion_dim)

        # 自适应权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(self.fusion_dim * 2, self.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.fusion_dim, 2),
            nn.Softmax(dim=-1)
        )

        # 输出投影
        self.output_proj = nn.Linear(self.fusion_dim, self.output_dim)

    def _build_simple_concat(self):
        """构建简单连接融合"""
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.clip_dim + self.geometry_dim, self.fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim, self.output_dim)
        )

    def forward(self, clip_features: torch.Tensor,
                geometry_features: torch.Tensor) -> torch.Tensor:
        """
        融合2D和3D特征

        Args:
            clip_features: CLIP 2D特征 [B, clip_dim]
            geometry_features: 几何3D特征 [B, geometry_dim]

        Returns:
            fused_features: 融合特征 [B, output_dim]
        """
        if self.fusion_type == "cross_attention":
            return self._cross_attention_forward(clip_features, geometry_features)
        elif self.fusion_type == "adaptive_fusion":
            return self._adaptive_fusion_forward(clip_features, geometry_features)
        elif self.fusion_type == "simple_concat":
            return self._simple_concat_forward(clip_features, geometry_features)

    # 投影
    def _cross_attention_forward(self, clip_feat, geometry_feat):
        clip_proj = self.clip_proj(clip_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        geometry_proj = self.geometry_proj(geometry_feat).unsqueeze(1)  # [B, 1, fusion_dim]

        # 交叉注意力：CLIP查询几何特征
        clip_enhanced, _ = self.cross_attention(clip_proj, geometry_proj, geometry_proj)

        # 交叉注意力：几何查询CLIP特征
        geometry_enhanced, _ = self.cross_attention(geometry_proj, clip_proj, clip_proj)

        # 连接和投影
        fused = torch.cat([clip_enhanced.squeeze(1), geometry_enhanced.squeeze(1)], dim=1)
        return self.output_proj(fused)

    def _adaptive_fusion_forward(self, clip_feat, geometry_feat):
        # 投影
        clip_proj = self.clip_proj(clip_feat)
        geometry_proj = self.geometry_proj(geometry_feat)

        # 计算自适应权重
        combined = torch.cat([clip_proj, geometry_proj], dim=1)
        weights = self.weight_net(combined)  # [B, 2]

        # 加权融合
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        fused = w1 * clip_proj + w2 * geometry_proj

        return self.output_proj(fused)

    def _simple_concat_forward(self, clip_feat, geometry_feat):
        # 简单连接
        concatenated = torch.cat([clip_feat, geometry_feat], dim=1)
        return self.fusion_proj(concatenated)


class AnomalyDetectionHead(nn.Module):
    """
    异常检测头
    基于融合特征进行异常检测
    """

    def __init__(self,
                 input_dim: int = 512,
                 hidden_dim: int = 256,
                 num_classes: int = 2,  # 正常/异常
                 detection_type: str = "classification"):
        super(AnomalyDetectionHead, self).__init__()

        self.detection_type = detection_type
        self.input_dim = input_dim

        if detection_type == "classification":
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes)
            )
        elif detection_type == "regression":
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # 异常分数 [0, 1]
            )
        else:
            raise ValueError(f"不支持的检测类型: {detection_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        异常检测前向传播

        Args:
            features: 融合特征 [B, input_dim]

        Returns:
            predictions: 预测结果
                - classification: [B, num_classes]
                - regression: [B, 1]
        """
        return self.head(features)


class GeoCLIP(nn.Module):
    """
    GeoCLIP主模型
    整合2D CLIP、3D几何、深度信息进行异常检测
    """

    def __init__(self,
                 # 2D模型配置
                 clip_model_name: str = "ViT-B/16",
                 clip_pretrained: str = "openai",

                 # 3D模型配置
                 depth_estimator_type: str = "DPT_Large",
                 geometry_encoder_config: Dict = None,

                 # 融合配置
                 fusion_type: str = "cross_attention",
                 fusion_dim: int = 1024,
                 output_dim: int = 512,

                 # 异常检测配置
                 detection_type: str = "regression",
                 num_classes: int = 2,

                 # 其他配置
                 device: str = "cuda",
                 freeze_clip: bool = False):

        super(GeoCLIP, self).__init__()

        self.device = device
        self.freeze_clip = freeze_clip

        # 1. 2D CLIP模型
        self.clip_model = self._load_clip_model(clip_model_name, clip_pretrained)
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # 2. 深度估计器
        self.depth_estimator = DepthEstimator(
            model_type=depth_estimator_type,
            device=device
        )

        # 3. 深度到体素转换器
        self.voxel_converter = DepthToVoxelConverter(
            voxel_size=64,
            depth_range=(0.1, 10.0)
        )

        # 4. 3D几何编码器
        if geometry_encoder_config is None:
            geometry_encoder_config = {
                'type': 'voxel',
                'in_channels': 4,  # RGB + Depth
                'output_channels': 512
            }

        self.geometry_encoder = create_geometry_encoder(geometry_encoder_config)

        # 5. 特征融合模块
        # 安全地获取CLIP维度
        def get_clip_dim(model):
            """安全地获取CLIP模型的特征维度"""
            # 方法1: 检查transformer.width
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'width'):
                return model.transformer.width

            # 方法2: 检查width属性
            if hasattr(model, 'width'):
                return model.width

            # 方法3: 检查visual.width
            if hasattr(model, 'visual') and hasattr(model.visual, 'width'):
                return model.visual.width

            # 方法4: 根据模型名称推断
            model_name_lower = clip_model_name.lower()
            if 'vit-b' in model_name_lower or 'vitb' in model_name_lower:
                return 768
            elif 'vit-l' in model_name_lower or 'vitl' in model_name_lower:
                return 1024
            elif 'vit-h' in model_name_lower or 'vith' in model_name_lower:
                return 1280
            elif 'rn50' in model_name_lower or 'resnet50' in model_name_lower:
                return 1024
            elif 'rn101' in model_name_lower or 'resnet101' in model_name_lower:
                return 512

            # 默认值
            print(f"⚠ 无法确定CLIP维度，使用默认值512")
            return 512

        # clip_dim = get_clip_dim(self.clip_model)
        clip_dim = 768
        geometry_dim = geometry_encoder_config['output_channels']

        print(f"  特征维度 - CLIP: {clip_dim}, 几何: {geometry_dim}")

        self.fusion_module = FeatureFusionModule(
            clip_dim=clip_dim,
            geometry_dim=geometry_dim,
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            fusion_type=fusion_type
        )
        clip_dim = self.clip_model.transformer.width if hasattr(self.clip_model, 'transformer') else 512
        geometry_dim = geometry_encoder_config['output_channels']

        self.fusion_module = FeatureFusionModule(
            clip_dim=clip_dim,
            geometry_dim=geometry_dim,
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            fusion_type=fusion_type
        )

        # 6. 异常检测头
        self.anomaly_head = AnomalyDetectionHead(
            input_dim=output_dim,
            detection_type=detection_type,
            num_classes=num_classes
        )

        print(f"GeoCLIP模型初始化完成:")
        print(f"  CLIP模型: {clip_model_name}")
        print(f"  深度估计: {depth_estimator_type}")
        print(f"  几何编码: {geometry_encoder_config['type']}")
        print(f"  融合方式: {fusion_type}")
        print(f"  检测类型: {detection_type}")

    # def _load_clip_model(self, model_name: str, pretrained: str):
    #     """加载CLIP模型 - 从本地缓存加载"""
    #
    #     import os
    #     from pathlib import Path
    #
    #     model_name_converted = model_name.replace('/', '-')
    #
    #     # 指定缓存目录
    #     cache_dir = Path.home() / '.cache' / 'open_clip'
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     print(f"正在从缓存加载CLIP模型: {model_name_converted}")
    #     print(f"缓存目录: {cache_dir}")
    #
    #     try:
    #         # 尝试从本地缓存加载
    #         model, _, preprocess = open_clip.create_model_and_transforms(
    #             model_name_converted,
    #             pretrained=pretrained if pretrained != 'openai' else 'openai',
    #             cache_dir=str(cache_dir)
    #         )
    #
    #         model = model.to(self.device)
    #         self.preprocess = preprocess
    #         print(f"成功加载预训练模型: {model_name_converted}")
    #         return model
    #
    #     except Exception as e:
    #         print(f"从缓存加载失败: {e}")
    #         # 检查缓存文件是否存在
    #         cache_files = list(cache_dir.glob("*.pt"))
    #         print(f"缓存目录中的文件: {cache_files}")
    #
    #         # 如果有.pt文件，尝试直接加载
    #         if cache_files:
    #             print("尝试直接加载本地权重文件...")
    #             model, _, preprocess = open_clip.create_model_and_transforms(
    #                 model_name_converted,
    #                 pretrained=None
    #             )
    #             # 手动加载权重
    #             state_dict = torch.load(cache_files[0], map_location=self.device)
    #             model.load_state_dict(state_dict, strict=False)
    #             model = model.to(self.device)
    #             self.preprocess = preprocess
    #             print("成功从本地文件加载权重")
    #             return model
    #         else:
    #             raise RuntimeError(f"缓存目录中没有找到模型文件: {cache_dir}")

    def _load_clip_model(self, model_name: str, pretrained: str):
        """加载CLIP模型 - 优先从本地缓存加载TorchScript模型"""
        import open_clip
        import torch
        from pathlib import Path
        from torchvision import transforms

        model_name_converted = model_name.replace('/', '-')
        cache_dir = Path.home() / '.cache' / 'open_clip'
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"正在从本地加载CLIP模型: {model_name_converted}")

        # 定义标准的CLIP预处理pipeline
        standard_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

        # 1. 优先尝试加载本地的TorchScript模型(.pt文件)
        pt_files = list(cache_dir.glob(f"*{model_name_converted}*.pt"))
        if not pt_files:
            pt_files = list(cache_dir.glob("*.pt"))

        if pt_files:
            print(f"发现本地TorchScript模型: {pt_files[0]}")
            try:
                jit_model = torch.jit.load(str(pt_files[0]), map_location=self.device)
                jit_model.eval()

                # 为TorchScript模型创建包装器,添加缺失的属性
                class TorchScriptWrapper:
                    def __init__(self, jit_model, device):
                        self.model = jit_model
                        self.device = device
                        # 尝试推断或设置默认属性
                        self.width = 768  # ViT-B默认宽度,可根据实际模型调整
                        self.visual = self  # 某些代码可能访问model.visual

                        # 添加transformer属性模拟
                        class TransformerProxy:
                            def __init__(self, width):
                                self.width = width

                        self.transformer = TransformerProxy(self.width)

                    def encode_image(self, image):
                        """图像编码方法"""
                        return self.model.encode_image(image)

                    def encode_text(self, text):
                        """文本编码方法"""
                        return self.model.encode_text(text)

                    def __call__(self, *args, **kwargs):
                        return self.model(*args, **kwargs)

                    def to(self, device):
                        self.model.to(device)
                        return self

                    def eval(self):
                        self.model.eval()
                        return self

                    def __getattr__(self, name):
                        # 转发其他属性访问到原始模型
                        try:
                            return getattr(self.model, name)
                        except AttributeError:
                            raise AttributeError(f"TorchScript模型没有属性: {name}")

                model = TorchScriptWrapper(jit_model, self.device)
                self.preprocess = standard_preprocess
                print("✓ 成功加载本地TorchScript模型(已添加兼容层)")
                return model
            except Exception as e:
                print(f"✗ 加载TorchScript模型失败: {e}")

        # 2. 尝试使用open_clip从本地缓存加载预训练权重
        try:
            print("尝试使用open_clip加载预训练模型...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name_converted,
                pretrained=pretrained if pretrained != 'openai' else 'openai',
                cache_dir=str(cache_dir)
            )
            model = model.to(self.device)
            model.eval()
            self.preprocess = preprocess
            print("✓ 成功加载open_clip预训练模型")
            return model
        except Exception as e:
            print(f"✗ open_clip加载失败: {e}")

        # 3. 最后回退: 创建无预训练权重的模型
        print("⚠ 回退到无预训练权重模型")
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name_converted,
                pretrained=None
            )
            model = model.to(self.device)
            model.eval()

            # 使用标准预处理或open_clip提供的预处理
            self.preprocess = preprocess if preprocess is not None else standard_preprocess
            print("✓ 成功创建无预训练权重模型(需要重新训练)")
            return model
        except Exception as e:
            print(f"✗ 创建模型失败: {e}")
            raise RuntimeError(f"无法加载或创建CLIP模型: {e}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        GeoCLIP前向传播

        Args:
            batch: 批次数据，包含：
                - 'image': RGB图像 [B, 3, H, W]
                - 'depth': 深度图 [B, 1, H, W] (可选，如果没有会自动估计)
                - 'text': 文本token (可选)

        Returns:
            输出字典，包含：
                - 'anomaly_score': 异常分数
                - 'clip_features': CLIP特征
                - 'geometry_features': 几何特征
                - 'fused_features': 融合特征
        """
        images = batch['image']
        batch_size = images.size(0)

        # 1. 获取深度图
        if 'depth' in batch and batch['depth'] is not None:
            depth_maps = batch['depth']
        else:
            # 使用深度估计器估计深度
            with torch.no_grad():
                depth_maps = self.depth_estimator(images)

        # 2. CLIP特征提取
        if 'text' in batch:
            # 如果有文本，使用图像-文本对比
            clip_features = self.clip_model.encode_image(images)
        else:
            # 仅使用图像编码
            clip_features = self.clip_model.encode_image(images)

        # 3. 转换为体素表示
        # 合并RGB和深度信息
        rgbd_images = torch.cat([images, depth_maps], dim=1)  # [B, 4, H, W]

        # 转换为3D体素
        voxels = self.voxel_converter.images_to_voxels(rgbd_images)  # [B, 4, D, H, W]

        # 4. 3D几何特征提取
        geometry_features = self.geometry_encoder(voxels)

        # 5. 特征融合
        fused_features = self.fusion_module(clip_features, geometry_features)

        # 6. 异常检测
        anomaly_predictions = self.anomaly_head(fused_features)

        # 返回结果
        results = {
            'anomaly_predictions': anomaly_predictions,
            'clip_features': clip_features,
            'geometry_features': geometry_features,
            'fused_features': fused_features,
            'depth_maps': depth_maps
        }

        return results

    def predict_anomaly(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        预测异常分数

        Args:
            batch: 输入批次

        Returns:
            anomaly_scores: 异常分数 [B] 或 [B, num_classes]
        """
        with torch.no_grad():
            results = self.forward(batch)
            predictions = results['anomaly_predictions']

            if self.anomaly_head.detection_type == "regression":
                return predictions.squeeze(-1)  # [B]
            else:
                # 分类情况，返回异常类别的概率
                return F.softmax(predictions, dim=-1)[:, 1]  # [B]

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'clip_frozen': self.freeze_clip,
            'fusion_type': self.fusion_module.fusion_type,
            'detection_type': self.anomaly_head.detection_type,
            'device': self.device
        }


def create_geoclip_model(config: Dict[str, Any]) -> GeoCLIP:
    """
    工厂函数：根据配置创建GeoCLIP模型

    Args:
        config: 模型配置字典

    Returns:
        GeoCLIP模型实例
    """
    return GeoCLIP(
        clip_model_name=config.get('clip_model', 'ViT-B/16'),
        clip_pretrained=config.get('clip_pretrained', 'openai'),
        depth_estimator_type=config.get('depth_estimator', 'DPT_Large'),
        geometry_encoder_config=config.get('geometry_encoder', None),
        fusion_type=config.get('fusion_type', 'cross_attention'),
        fusion_dim=config.get('fusion_dim', 1024),
        output_dim=config.get('output_dim', 512),
        detection_type=config.get('detection_type', 'regression'),
        num_classes=config.get('num_classes', 2),
        device=config.get('device', 'cuda'),
        freeze_clip=config.get('freeze_clip', False)
    )


# 测试代码
if __name__ == "__main__":
    print("=== 测试GeoCLIP模型 ===")

    try:
        # 创建模型配置
        config = {
            'clip_model': 'ViT-B/16',
            'depth_estimator': 'DPT_Large',
            'fusion_type': 'cross_attention',
            'detection_type': 'regression',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

        # 创建模型
        model = create_geoclip_model(config)
        print(f"✅ 模型创建成功")
        print(f"模型信息: {model.get_model_info()}")

        # 创建测试数据
        batch_size = 2
        test_batch = {
            'image': torch.randn(batch_size, 3, 224, 224),
            # 'depth': torch.randn(batch_size, 1, 224, 224),  # 可选
        }

        # 移动到设备
        device = config['device']
        model = model.to(device)
        for key in test_batch:
            test_batch[key] = test_batch[key].to(device)

        # 前向传播测试
        print("\n测试前向传播...")
        results = model(test_batch)

        print(f"✅ 前向传播成功:")
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")

        # 异常预测测试
        anomaly_scores = model.predict_anomaly(test_batch)
        print(f"✅ 异常预测: {anomaly_scores.shape}")
        print(f"异常分数范围: {anomaly_scores.min():.3f} - {anomaly_scores.max():.3f}")

        print("\n🎉 GeoCLIP模型测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()