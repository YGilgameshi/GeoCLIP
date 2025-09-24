"""
GeoCLIP - 特征融合模块
2D CLIP特征与3D几何特征融合策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math


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
        elif fusion_type == "multihead_attention":
            self._build_multihead_attention()
        elif fusion_type == "gated_fusion":
            self._build_gated_fusion()
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

    def _build_multihead_attention(self):
        """构建多头注意力融合"""
        # 投影层
        self.clip_proj = nn.Linear(self.clip_dim, self.fusion_dim)
        self.geometry_proj = nn.Linear(self.geometry_dim, self.fusion_dim)

        # 多头自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.fusion_dim * 2, self.output_dim)
        )

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(self.fusion_dim)
        self.layer_norm2 = nn.LayerNorm(self.output_dim)

    def _build_gated_fusion(self):
        """构建门控融合"""
        # 特征投影
        self.clip_proj = nn.Linear(self.clip_dim, self.fusion_dim)
        self.geometry_proj = nn.Linear(self.geometry_dim, self.fusion_dim)

        # 门控网络
        self.gate_clip = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Sigmoid()
        )

        self.gate_geometry = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.Sigmoid()
        )

        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim),
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
        elif self.fusion_type == "multihead_attention":
            return self._multihead_attention_forward(clip_features, geometry_features)
        elif self.fusion_type == "gated_fusion":
            return self._gated_fusion_forward(clip_features, geometry_features)

    def _cross_attention_forward(self, clip_feat, geometry_feat):
        """交叉注意力前向传播"""
        # 投影
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
        """自适应融合前向传播"""
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
        """简单连接前向传播"""
        # 简单连接
        concatenated = torch.cat([clip_feat, geometry_feat], dim=1)
        return self.fusion_proj(concatenated)

    def _multihead_attention_forward(self, clip_feat, geometry_feat):
        """多头注意力前向传播"""
        # 投影
        clip_proj = self.clip_proj(clip_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        geometry_proj = self.geometry_proj(geometry_feat).unsqueeze(1)  # [B, 1, fusion_dim]

        # 连接特征
        combined = torch.cat([clip_proj, geometry_proj], dim=1)  # [B, 2, fusion_dim]

        # 自注意力
        attended, _ = self.self_attention(combined, combined, combined)
        attended = self.layer_norm1(attended + combined)  # 残差连接

        # 全局平均池化
        pooled = attended.mean(dim=1)  # [B, fusion_dim]

        # 前馈网络
        output = self.ffn(pooled)
        output = self.layer_norm2(output)

        return output

    def _gated_fusion_forward(self, clip_feat, geometry_feat):
        """门控融合前向传播"""
        # 投影
        clip_proj = self.clip_proj(clip_feat)
        geometry_proj = self.geometry_proj(geometry_feat)

        # 计算门控
        clip_gate = self.gate_clip(clip_proj)
        geometry_gate = self.gate_geometry(geometry_proj)

        # 门控融合
        gated_clip = clip_gate * clip_proj
        gated_geometry = geometry_gate * geometry_proj

        # 相加并通过融合网络
        fused = gated_clip + gated_geometry
        return self.fusion_net(fused)

    def get_fusion_info(self) -> Dict[str, Any]:
        """获取融合模块信息"""
        return {
            'fusion_type': self.fusion_type,
            'clip_dim': self.clip_dim,
            'geometry_dim': self.geometry_dim,
            'fusion_dim': self.fusion_dim,
            'output_dim': self.output_dim,
            'parameters': sum(p.numel() for p in self.parameters())
        }


class BilinearFusionModule(nn.Module):
    """
    双线性融合模块
    使用双线性变换进行特征融合
    """

    def __init__(self,
                 clip_dim: int = 512,
                 geometry_dim: int = 512,
                 output_dim: int = 512,
                 use_dropout: bool = True):
        super(BilinearFusionModule, self).__init__()

        self.clip_dim = clip_dim
        self.geometry_dim = geometry_dim
        self.output_dim = output_dim

        # 双线性变换
        self.bilinear = nn.Bilinear(clip_dim, geometry_dim, output_dim)

        # 单独的线性变换
        self.clip_linear = nn.Linear(clip_dim, output_dim)
        self.geometry_linear = nn.Linear(geometry_dim, output_dim)

        # 输出层
        self.output_proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.1) if use_dropout else nn.Identity(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, clip_features: torch.Tensor,
                geometry_features: torch.Tensor) -> torch.Tensor:
        """
        双线性融合前向传播

        Args:
            clip_features: CLIP特征 [B, clip_dim]
            geometry_features: 几何特征 [B, geometry_dim]

        Returns:
            fused_features: 融合特征 [B, output_dim]
        """
        # 双线性项
        bilinear_term = self.bilinear(clip_features, geometry_features)

        # 线性项
        clip_term = self.clip_linear(clip_features)
        geometry_term = self.geometry_linear(geometry_features)

        # 融合
        fused = bilinear_term + clip_term + geometry_term

        return self.output_proj(fused)


class ResidualFusionModule(nn.Module):
    """
    残差融合模块
    使用残差连接进行特征融合
    """

    def __init__(self,
                 clip_dim: int = 512,
                 geometry_dim: int = 512,
                 hidden_dim: int = 1024,
                 output_dim: int = 512,
                 num_layers: int = 3):
        super(ResidualFusionModule, self).__init__()

        # 输入投影
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)
        self.geometry_proj = nn.Linear(geometry_dim, hidden_dim)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _make_residual_block(self, dim: int) -> nn.Module:
        """创建残差块"""
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

    def forward(self, clip_features: torch.Tensor,
                geometry_features: torch.Tensor) -> torch.Tensor:
        """
        残差融合前向传播
        """
        # 投影
        clip_proj = self.clip_proj(clip_features)
        geometry_proj = self.geometry_proj(geometry_features)

        # 初始融合
        x = clip_proj + geometry_proj

        # 残差块
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual  # 残差连接

        # 输出投影
        return self.output_proj(x)


def create_fusion_module(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建融合模块

    Args:
        config: 融合模块配置

    Returns:
        fusion_module: 融合模块实例
    """
    fusion_type = config.get('type', 'cross_attention')

    if fusion_type in ['cross_attention', 'adaptive_fusion', 'simple_concat',
                       'multihead_attention', 'gated_fusion']:
        return FeatureFusionModule(
            clip_dim=config.get('clip_dim', 512),
            geometry_dim=config.get('geometry_dim', 512),
            fusion_dim=config.get('fusion_dim', 1024),
            output_dim=config.get('output_dim', 512),
            fusion_type=fusion_type
        )
    elif fusion_type == 'bilinear':
        return BilinearFusionModule(
            clip_dim=config.get('clip_dim', 512),
            geometry_dim=config.get('geometry_dim', 512),
            output_dim=config.get('output_dim', 512),
            use_dropout=config.get('use_dropout', True)
        )
    elif fusion_type == 'residual':
        return ResidualFusionModule(
            clip_dim=config.get('clip_dim', 512),
            geometry_dim=config.get('geometry_dim', 512),
            hidden_dim=config.get('hidden_dim', 1024),
            output_dim=config.get('output_dim', 512),
            num_layers=config.get('num_layers', 3)
        )
    else:
        raise ValueError(f"不支持的融合类型: {fusion_type}")


# 测试函数
def test_fusion_modules():
    """测试融合模块"""
    print("=== 测试特征融合模块 ===")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 4
        clip_dim = 512
        geometry_dim = 512

        # 创建测试数据
        clip_features = torch.randn(batch_size, clip_dim, device=device)
        geometry_features = torch.randn(batch_size, geometry_dim, device=device)

        # 测试所有融合类型
        fusion_types = [
            'cross_attention', 'adaptive_fusion', 'simple_concat',
            'multihead_attention', 'gated_fusion', 'bilinear', 'residual'
        ]

        for fusion_type in fusion_types:
            print(f"\n测试 {fusion_type} 融合:")

            config = {
                'type': fusion_type,
                'clip_dim': clip_dim,
                'geometry_dim': geometry_dim,
                'fusion_dim': 1024,
                'output_dim': 512
            }

            fusion_module = create_fusion_module(config).to(device)

            # 前向传播
            with torch.no_grad():
                fused_features = fusion_module(clip_features, geometry_features)

            print(f"✅ {fusion_type}: {clip_features.shape} + {geometry_features.shape} -> {fused_features.shape}")

            # 获取模块信息
            if hasattr(fusion_module, 'get_fusion_info'):
                info = fusion_module.get_fusion_info()
                print(f"   参数量: {info['parameters']:,}")

        print("\n🎉 融合模块测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fusion_modules()