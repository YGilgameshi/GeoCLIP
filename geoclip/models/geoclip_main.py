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
    def __init__(self,
                 clip_model_name: str = "ViT-B/16",
                 clip_pretrained: str = "openai",
                 depth_estimator_type: str = "DPT_Large",
                 geometry_encoder_config: Dict = None,
                 fusion_type: str = "cross_attention",
                 fusion_dim: int = 1024,
                 output_dim: int = 512,
                 detection_type: str = "regression",
                 num_classes: int = 2,
                 device: str = "cuda",
                 freeze_clip: bool = False):

        super(GeoCLIP, self).__init__()

        # ========== 关键：统一设备管理 ==========
        self.device = torch.device(device)
        self.freeze_clip = freeze_clip

        print(f"🔧 GeoCLIP初始化，目标设备: {self.device}")

        # 1. CLIP模型（可能固定在cuda:0）
        print("1️⃣ 加载CLIP模型...")
        self.clip_model = self._load_clip_model(clip_model_name, clip_pretrained)
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # 检测CLIP实际设备
        if hasattr(self.clip_model, 'device'):
            clip_device = self.clip_model.device
            print(f"   CLIP设备: {clip_device}")

        # 检测CLIP输入尺寸
        if hasattr(self.clip_model, 'input_size'):
            self.clip_input_size = self.clip_model.input_size
        else:
            self.clip_input_size = 224
        print(f"   CLIP输入尺寸: {self.clip_input_size}x{self.clip_input_size}")

        # 2. 深度估计器 - 明确设置设备
        print("2️⃣ 加载深度估计器...")
        self.depth_input_size = 384
        self.depth_estimator = DepthEstimator(
            model_type=depth_estimator_type,
            device=str(self.device),  # 确保传递字符串格式
            input_size=(self.depth_input_size, self.depth_input_size)
        )
        # 确保深度估计器在正确设备上
        self.depth_estimator = self.depth_estimator.to(self.device)
        print(f"   深度估计器设备: {next(self.depth_estimator.parameters()).device}")

        # 3. 体素转换器 - 不是nn.Module，需要手动管理设备
        print("3️⃣ 初始化体素转换器...")
        from geoclip.utils.voxel_utils import DepthToVoxelConverter
        self.voxel_converter = DepthToVoxelConverter(
            voxel_size=64,
            depth_range=(0.1, 10.0),
            use_color=True
        )
        # 体素转换器是工具类，在forward中处理设备

        # 4. 几何编码器 - 确保在正确设备上
        print("4️⃣ 初始化几何编码器...")
        if geometry_encoder_config is None:
            geometry_encoder_config = {
                'type': 'voxel',
                'in_channels': 4,
                'output_channels': 512
            }

        from geoclip.models.geometry_encoder import create_geometry_encoder
        self.geometry_encoder = create_geometry_encoder(geometry_encoder_config)
        self.geometry_encoder = self.geometry_encoder.to(self.device)
        print(f"   几何编码器设备: {next(self.geometry_encoder.parameters()).device}")

        # 5. 特征融合模块 - 确保在正确设备上
        print("5️⃣ 初始化特征融合模块...")
        clip_dim = 512
        geometry_dim = geometry_encoder_config['output_channels']

        from geoclip.models.fusion_module import FeatureFusionModule
        self.fusion_module = FeatureFusionModule(
            clip_dim=clip_dim,
            geometry_dim=geometry_dim,
            fusion_dim=fusion_dim,
            output_dim=output_dim,
            fusion_type=fusion_type
        )
        self.fusion_module = self.fusion_module.to(self.device)
        print(f"   融合模块设备: {next(self.fusion_module.parameters()).device}")

        # 6. 异常检测头 - 确保在正确设备上
        print("6️⃣ 初始化异常检测头...")

        self.anomaly_head = AnomalyDetectionHead(
            input_dim=output_dim,
            detection_type=detection_type,
            num_classes=num_classes
        )
        self.anomaly_head = self.anomaly_head.to(self.device)
        print(f"   检测头设备: {next(self.anomaly_head.parameters()).device}")

        print(f"✅ GeoCLIP模型初始化完成")
        print(f"   特征维度 - CLIP: {clip_dim}, 几何: {geometry_dim}")

        print(f"GeoCLIP模型初始化完成:")
        print(f"  CLIP模型: {clip_model_name}")
        print(f"  深度估计: {depth_estimator_type}")
        print(f"  几何编码: {geometry_encoder_config['type']}")
        print(f"  融合方式: {fusion_type}")
        print(f"  检测类型: {detection_type}")


    # def _load_clip_model(self, model_name: str, pretrained: str):
    #     """
    #     加载CLIP模型 - 优先从本地缓存加载TorchScript模型
    #     注意: TorchScript模型固定在cuda:0上
    #     """
    #     import open_clip
    #     import torch
    #     from pathlib import Path
    #     from torchvision import transforms
    #
    #     model_name_converted = model_name.replace('/', '-')
    #     cache_dir = Path.home() / '.cache' / 'open_clip'
    #     cache_dir.mkdir(parents=True, exist_ok=True)
    #
    #     print(f"正在从本地加载CLIP模型: {model_name_converted}")
    #
    #     # 定义创建预处理的函数
    #     def create_preprocess(size):
    #         return transforms.Compose([
    #             transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    #             transforms.CenterCrop(size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.48145466, 0.4578275, 0.40821073],
    #                 std=[0.26862954, 0.26130258, 0.27577711]
    #             )
    #         ])
    #
    #     # 初始默认预处理
    #     standard_preprocess = create_preprocess(224)
    #
    #     # ========== 1. 优先加载本地TorchScript模型 ==========
    #     pt_files = list(cache_dir.glob(f"*{model_name_converted}*.pt"))
    #     if not pt_files:
    #         pt_files = list(cache_dir.glob("*.pt"))
    #
    #     if pt_files:
    #         print(f"发现本地TorchScript模型: {pt_files[0]}")
    #
    #         # 检查GPU可用性
    #         if not torch.cuda.is_available():
    #             print("❌ TorchScript模型需要GPU，但系统无GPU可用")
    #             print("跳过TorchScript加载，尝试其他方式...")
    #         else:
    #             try:
    #                 print("⚠️ TorchScript模型固定在cuda:0")
    #                 print("正在加载模型到cuda:0...")
    #
    #                 # 强制加载到cuda:0
    #                 jit_model = torch.jit.load(str(pt_files[0]), map_location='cuda:0')
    #                 jit_model = jit_model.cuda(0)
    #                 jit_model.eval()
    #
    #                 print("✓ TorchScript模型已加载到cuda:0")
    #
    #                 # ========== 检测模型输入尺寸 ==========
    #                 input_size = 224  # 默认值
    #                 print("正在检测模型输入尺寸...")
    #
    #                 # 尝试224
    #                 try:
    #                     test_input = torch.randn(1, 3, 224, 224).cuda(0)
    #                     with torch.no_grad():
    #                         jit_model.encode_image(test_input)
    #                     input_size = 224
    #                     print("✓ 检测到模型输入尺寸: 224x224")
    #                 except Exception as e1:
    #                     # 尝试336
    #                     try:
    #                         test_input = torch.randn(1, 3, 336, 336).cuda(0)
    #                         with torch.no_grad():
    #                             jit_model.encode_image(test_input)
    #                         input_size = 336
    #                         print("✓ 检测到模型输入尺寸: 336x336")
    #                     except Exception as e2:
    #                         # 尝试384
    #                         try:
    #                             test_input = torch.randn(1, 3, 384, 384).cuda(0)
    #                             with torch.no_grad():
    #                                 jit_model.encode_image(test_input)
    #                             input_size = 384
    #                             print("✓ 检测到模型输入尺寸: 384x384")
    #                         except Exception as e3:
    #                             print(f"⚠️ 无法自动检测输入尺寸")
    #                             print(f"  224错误: {str(e1)[:100]}")
    #                             print(f"  336错误: {str(e2)[:100]}")
    #                             print(f"  384错误: {str(e3)[:100]}")
    #                             print("使用默认224")
    #                             input_size = 224
    #
    #                 # 根据检测到的尺寸创建预处理
    #                 standard_preprocess = create_preprocess(input_size)
    #
    #                 # ========== TorchScript包装器 ==========
    #                 class TorchScriptWrapper(torch.nn.Module):
    #                     """TorchScript模型包装器 - 固定cuda:0"""
    #
    #                     def __init__(self, jit_model, input_size):
    #                         super().__init__()
    #                         self.model = jit_model
    #                         self.device = torch.device('cuda:0')
    #                         self.input_size = input_size
    #                         self.width = 768
    #                         self.visual = self
    #
    #                         class TransformerProxy:
    #                             def __init__(self, width):
    #                                 self.width = width
    #
    #                         self.transformer = TransformerProxy(self.width)
    #
    #                     def encode_image(self, image):
    #                         """图像编码 - 自动处理设备和尺寸"""
    #                         # 1. 确保在cuda:0
    #                         if not image.is_cuda:
    #                             image = image.cuda(0)
    #                         elif image.device.index != 0:
    #                             image = image.cuda(0)
    #
    #                         # 2. 检查并调整尺寸（关键修复）
    #                         current_size = image.shape[-2:]
    #                         if current_size != (self.input_size, self.input_size):
    #                             print(f"⚠️ 输入尺寸{current_size}不匹配，期望{self.input_size}x{self.input_size}")
    #                             print(f"   这说明预处理没有正确应用，请检查数据加载流程")
    #                             # 紧急resize
    #                             import torch.nn.functional as F
    #                             image = F.interpolate(
    #                                 image,
    #                                 size=(self.input_size, self.input_size),
    #                                 mode='bicubic',
    #                                 align_corners=False
    #                             )
    #                             print(f"✓ 已紧急调整到{self.input_size}x{self.input_size}")
    #
    #                         # 3. 调用模型
    #                         try:
    #                             if hasattr(self.model, 'encode_image'):
    #                                 return self.model.encode_image(image)
    #                             else:
    #                                 return self.model(image)
    #                         except RuntimeError as e:
    #                             error_msg = str(e)
    #                             if "577" in error_msg and "197" in error_msg:
    #                                 print(f"❌ Token数量不匹配错误!")
    #                                 print(f"   这意味着输入尺寸仍然不正确")
    #                                 print(f"   期望输入: {self.input_size}x{self.input_size}")
    #                                 print(f"   实际输入: {image.shape}")
    #                                 print(f"   模型可能在224x224上训练，但收到了更大的输入")
    #                             raise
    #
    #                     def encode_text(self, text):
    #                         """文本编码"""
    #                         if not text.is_cuda:
    #                             text = text.cuda(0)
    #                         elif text.device.index != 0:
    #                             text = text.cuda(0)
    #
    #                         if hasattr(self.model, 'encode_text'):
    #                             return self.model.encode_text(text)
    #                         else:
    #                             raise AttributeError("TorchScript模型没有encode_text方法")
    #
    #                     def forward(self, image):
    #                         return self.encode_image(image)
    #
    #                     def __call__(self, *args, **kwargs):
    #                         if len(args) == 1 and isinstance(args[0], torch.Tensor):
    #                             return self.forward(args[0])
    #                         return self.model(*args, **kwargs)
    #
    #                     def to(self, device):
    #                         """固定cuda:0，忽略其他请求"""
    #                         if str(device) not in ['cuda:0', 'cuda']:
    #                             print(f"⚠️ TorchScript模型固定在cuda:0，忽略to({device})")
    #                         return self
    #
    #                     def cuda(self, device=None):
    #                         if device is not None and device != 0:
    #                             print(f"⚠️ TorchScript模型固定在cuda:0，忽略cuda({device})")
    #                         return self
    #
    #                     def cpu(self):
    #                         print(f"⚠️ TorchScript模型固定在cuda:0，不支持转到CPU")
    #                         return self
    #
    #                     def eval(self):
    #                         self.model.eval()
    #                         return self
    #
    #                     def train(self, mode=True):
    #                         if mode:
    #                             self.model.train()
    #                         else:
    #                             self.model.eval()
    #                         return self
    #
    #                     def parameters(self):
    #                         return self.model.parameters()
    #
    #                     def __getattr__(self, name):
    #                         try:
    #                             return super().__getattr__(name)
    #                         except AttributeError:
    #                             try:
    #                                 return getattr(self.model, name)
    #                             except AttributeError:
    #                                 raise AttributeError(f"TorchScript模型没有属性: {name}")
    #
    #                 # 创建包装器
    #                 model = TorchScriptWrapper(jit_model, input_size)
    #                 self.preprocess = standard_preprocess
    #
    #                 print("✓ 成功加载TorchScript模型")
    #                 print(f"✓ 模型固定在: cuda:0")
    #                 print(f"✓ 输入尺寸: {input_size}x{input_size}")
    #                 print(f"✓ 特征维度: {model.width}")
    #                 print(f"⚠️ 重要: 请确保数据预处理使用{input_size}x{input_size}尺寸")
    #
    #                 return model
    #
    #             except Exception as e:
    #                 print(f"✗ 加载TorchScript模型失败: {e}")
    #                 import traceback
    #                 traceback.print_exc()
    #                 print("尝试其他加载方式...")
    #
    #     # ========== 2. 使用open_clip加载 ==========
    #     try:
    #         print("尝试使用open_clip加载预训练模型...")
    #         model, _, preprocess = open_clip.create_model_and_transforms(
    #             model_name_converted,
    #             pretrained=pretrained if pretrained != 'openai' else 'openai',
    #             cache_dir=str(cache_dir)
    #         )
    #         model = model.to(self.device)
    #         model.eval()
    #         self.preprocess = preprocess
    #         print(f"✓ 成功加载open_clip预训练模型")
    #         return model
    #     except Exception as e:
    #         print(f"✗ open_clip加载失败: {e}")
    #
    #     # ========== 3. 创建无预训练模型 ==========
    #     print("⚠️ 回退到无预训练权重模型")
    #     try:
    #         model, _, preprocess = open_clip.create_model_and_transforms(
    #             model_name_converted,
    #             pretrained=None
    #         )
    #         model = model.to(self.device)
    #         model.eval()
    #         self.preprocess = preprocess if preprocess is not None else standard_preprocess
    #         print(f"✓ 成功创建无预训练权重模型")
    #         print("⚠️ 此模型需要重新训练")
    #         return model
    #     except Exception as e:
    #         print(f"✗ 创建模型失败: {e}")
    #         raise RuntimeError(f"无法加载或创建CLIP模型: {e}")

    def _load_clip_model(self, model_name: str, pretrained: str):
        """
        加载CLIP模型 - 优先从本地缓存加载TorchScript模型
        注意: TorchScript模型固定在cuda:0上
        """
        import open_clip
        import torch
        from pathlib import Path
        from torchvision import transforms

        model_name_converted = model_name.replace('/', '-')
        cache_dir = Path.home() / '.cache' / 'open_clip'
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"正在从本地加载CLIP模型: {model_name_converted}")

        # 定义创建预处理的函数
        def create_preprocess(size):
            return transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])

        # 初始默认预处理
        standard_preprocess = create_preprocess(224)

        # ========== 1. 优先加载本地TorchScript模型 ==========
        pt_files = list(cache_dir.glob(f"*{model_name_converted}*.pt"))
        if not pt_files:
            pt_files = list(cache_dir.glob("*.pt"))

        if pt_files:
            print(f"发现本地TorchScript模型: {pt_files[0]}")

            # 检查GPU可用性
            if not torch.cuda.is_available():
                print("❌ TorchScript模型需要GPU，但系统无GPU可用")
                print("跳过TorchScript加载，尝试其他方式...")
            else:
                try:
                    print("⚠️ TorchScript模型固定在cuda:0")
                    print("正在加载模型到cuda:0...")

                    # 强制加载到cuda:0
                    jit_model = torch.jit.load(str(pt_files[0]), map_location='cuda:0')
                    jit_model = jit_model.cuda(0)
                    jit_model.eval()

                    print("✓ TorchScript模型已加载到cuda:0")

                    # ========== 检测模型输入尺寸 ==========
                    input_size = 224  # 默认值
                    print("正在检测模型输入尺寸...")

                    # 尝试224
                    try:
                        test_input = torch.randn(1, 3, 224, 224).cuda(0)
                        with torch.no_grad():
                            jit_model.encode_image(test_input)
                        input_size = 224
                        print("✓ 检测到模型输入尺寸: 224x224")
                    except Exception as e1:
                        # 尝试336
                        try:
                            test_input = torch.randn(1, 3, 336, 336).cuda(0)
                            with torch.no_grad():
                                jit_model.encode_image(test_input)
                            input_size = 336
                            print("✓ 检测到模型输入尺寸: 336x336")
                        except Exception as e2:
                            # 尝试384
                            try:
                                test_input = torch.randn(1, 3, 384, 384).cuda(0)
                                with torch.no_grad():
                                    jit_model.encode_image(test_input)
                                input_size = 384
                                print("✓ 检测到模型输入尺寸: 384x384")
                            except Exception as e3:
                                print(f"⚠️ 无法自动检测输入尺寸")
                                print(f"  224错误: {str(e1)[:100]}")
                                print(f"  336错误: {str(e2)[:100]}")
                                print(f"  384错误: {str(e3)[:100]}")
                                print("使用默认224")
                                input_size = 224

                    # 根据检测到的尺寸创建预处理
                    standard_preprocess = create_preprocess(input_size)

                    # ========== TorchScript包装器 ==========
                    class TorchScriptWrapper(torch.nn.Module):
                        """TorchScript模型包装器 - 固定cuda:0"""

                        def __init__(self, jit_model, input_size):
                            super().__init__()
                            self.model = jit_model
                            self.device = torch.device('cuda:0')
                            self.input_size = input_size
                            self.width = 768
                            self.visual = self

                            class TransformerProxy:
                                def __init__(self, width):
                                    self.width = width

                            self.transformer = TransformerProxy(self.width)

                        def encode_image(self, image):
                            """图像编码 - 自动处理设备和尺寸"""
                            # 1. 确保在cuda:0
                            if not image.is_cuda:
                                image = image.cuda(0)
                            elif image.device.index != 0:
                                image = image.cuda(0)

                            # 2. 检查并调整尺寸（关键修复）
                            current_size = image.shape[-2:]
                            if current_size != (self.input_size, self.input_size):
                                print(f"⚠️ 输入尺寸{current_size}不匹配，期望{self.input_size}x{self.input_size}")
                                print(f"   这说明预处理没有正确应用，请检查数据加载流程")
                                # 紧急resize
                                import torch.nn.functional as F
                                image = F.interpolate(
                                    image,
                                    size=(self.input_size, self.input_size),
                                    mode='bicubic',
                                    align_corners=False
                                )
                                print(f"✓ 已紧急调整到{self.input_size}x{self.input_size}")

                            # 3. 调用模型
                            try:
                                if hasattr(self.model, 'encode_image'):
                                    return self.model.encode_image(image)
                                else:
                                    return self.model(image)
                            except RuntimeError as e:
                                error_msg = str(e)
                                if "577" in error_msg and "197" in error_msg:
                                    print(f"❌ Token数量不匹配错误!")
                                    print(f"   这意味着输入尺寸仍然不正确")
                                    print(f"   期望输入: {self.input_size}x{self.input_size}")
                                    print(f"   实际输入: {image.shape}")
                                    print(f"   模型可能在224x224上训练，但收到了更大的输入")
                                raise

                        def encode_text(self, text):
                            """文本编码"""
                            if not text.is_cuda:
                                text = text.cuda(0)
                            elif text.device.index != 0:
                                text = text.cuda(0)

                            if hasattr(self.model, 'encode_text'):
                                return self.model.encode_text(text)
                            else:
                                raise AttributeError("TorchScript模型没有encode_text方法")

                        def forward(self, image):
                            return self.encode_image(image)

                        def __call__(self, *args, **kwargs):
                            if len(args) == 1 and isinstance(args[0], torch.Tensor):
                                return self.forward(args[0])
                            return self.model(*args, **kwargs)

                        def to(self, device):
                            """固定cuda:0，忽略其他请求"""
                            if str(device) not in ['cuda:0', 'cuda']:
                                print(f"⚠️ TorchScript模型固定在cuda:0，忽略to({device})")
                            return self

                        def cuda(self, device=None):
                            if device is not None and device != 0:
                                print(f"⚠️ TorchScript模型固定在cuda:0，忽略cuda({device})")
                            return self

                        def cpu(self):
                            print(f"⚠️ TorchScript模型固定在cuda:0，不支持转到CPU")
                            return self

                        def eval(self):
                            self.model.eval()
                            return self

                        def train(self, mode=True):
                            if mode:
                                self.model.train()
                            else:
                                self.model.eval()
                            return self

                        def parameters(self):
                            return self.model.parameters()

                        def __getattr__(self, name):
                            try:
                                return super().__getattr__(name)
                            except AttributeError:
                                try:
                                    return getattr(self.model, name)
                                except AttributeError:
                                    raise AttributeError(f"TorchScript模型没有属性: {name}")

                    # 创建包装器
                    model = TorchScriptWrapper(jit_model, input_size)
                    self.preprocess = standard_preprocess

                    print("✓ 成功加载TorchScript模型")
                    print(f"✓ 模型固定在: cuda:0")
                    print(f"✓ 输入尺寸: {input_size}x{input_size}")
                    print(f"✓ 特征维度: {model.width}")
                    print(f"⚠️ 重要: 请确保数据预处理使用{input_size}x{input_size}尺寸")

                    return model

                except Exception as e:
                    print(f"✗ 加载TorchScript模型失败: {e}")
                    import traceback
                    traceback.print_exc()
                    print("尝试其他加载方式...")

        # ========== 2. 使用open_clip加载 ==========
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
            print(f"✓ 成功加载open_clip预训练模型")
            return model
        except Exception as e:
            print(f"✗ open_clip加载失败: {e}")

        # ========== 3. 创建无预训练模型 ==========
        print("⚠️ 回退到无预训练权重模型")
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name_converted,
                pretrained=None
            )
            model = model.to(self.device)
            model.eval()
            self.preprocess = preprocess if preprocess is not None else standard_preprocess
            print(f"✓ 成功创建无预训练权重模型")
            print("⚠️ 此模型需要重新训练")
            return model
        except Exception as e:
            print(f"✗ 创建模型失败: {e}")
            raise RuntimeError(f"无法加载或创建CLIP模型: {e}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 确保所有数据在正确设备上
        """
        # ========== 设备检查和转换 ==========
        images = batch['image']

        # 关键：确保输入在正确设备上
        if images.device != self.device:
            print(f"⚠️ 输入图像设备不匹配: {images.device} -> {self.device}")
            images = images.to(self.device)

        batch_size = images.size(0)
        original_size = images.shape[-2:]

        # ========== 1. CLIP特征提取 ==========
        # CLIP可能固定在cuda:0，需要特殊处理
        if hasattr(self.clip_model, 'device'):
            clip_device = self.clip_model.device
            if images.device != clip_device:
                images_for_clip = images.to(clip_device)
            else:
                images_for_clip = images
        else:
            images_for_clip = images

        clip_features = self.clip_model.encode_image(images_for_clip)  # [B, 768]
        if clip_features.dtype == torch.float16:
            clip_features = clip_features.float()

        # 确保CLIP输出在正确设备上
        if clip_features.device != self.device:
            clip_features = clip_features.to(self.device)

        # ========== 2. 深度估计 ==========
        if 'depth' in batch and batch['depth'] is not None:
            depth_maps = batch['depth'].to(self.device)
            if depth_maps.shape[-2:] != original_size:
                depth_maps = F.interpolate(
                    depth_maps,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
        else:
            with torch.no_grad():
                # 为深度估计准备输入（可能需要resize到384）
                if original_size[0] != self.depth_input_size:
                    images_for_depth = F.interpolate(
                        images,
                        size=(self.depth_input_size, self.depth_input_size),
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    images_for_depth = images

                # 深度估计
                depth_maps = self.depth_estimator(images_for_depth)

                # Resize回原始尺寸
                if depth_maps.shape[-2:] != original_size:
                    depth_maps = F.interpolate(
                        depth_maps,
                        size=original_size,
                        mode='bilinear',
                        align_corners=False
                    )

        # 确保深度图在正确设备上
        depth_maps = depth_maps.to(self.device)

        # ========== 3. 合并RGBD ==========
        rgbd_images = torch.cat([images, depth_maps], dim=1)  # [B, 4, H, W]

        # ========== 4. 体素转换 ==========
        # 体素转换器是工具类，确保输入在正确设备上
        voxels = self.voxel_converter.images_to_voxels(rgbd_images)
        voxels = voxels.to(self.device)  # 确保输出在正确设备上

        # ========== 5. 几何特征提取 ==========
        geometry_features = self.geometry_encoder(voxels)

        # ========== 6. 特征融合 ==========
        fused_features = self.fusion_module(clip_features, geometry_features)

        # ========== 7. 异常检测 ==========
        anomaly_predictions = self.anomaly_head(fused_features)

        return {
            'anomaly_predictions': anomaly_predictions,
            'clip_features': clip_features,
            'geometry_features': geometry_features,
            'fused_features': fused_features,
            'depth_maps': depth_maps
        }

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