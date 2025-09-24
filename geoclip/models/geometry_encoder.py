
"""
GeoCLIP - 3D几何编码器
处理3D体素数据的深度网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict, Union


class Conv3DBlock(nn.Module):
    """3D卷积块"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 use_bn: bool = True,
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super(Conv3DBlock, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not use_bn)

        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResNet3DBlock(nn.Module):
    """3D ResNet块"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(ResNet3DBlock, self).__init__()

        self.conv1 = Conv3DBlock(in_channels, out_channels, kernel_size=3,
                                 stride=stride, padding=1)
        self.conv2 = Conv3DBlock(out_channels, out_channels, kernel_size=3,
                                 stride=1, padding=1, activation='none')

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VoxelEncoder(nn.Module):
    """
    基础体素编码器
    使用3D CNN提取体素特征
    """

    def __init__(self,
                 in_channels: int = 3,
                 base_channels: int = 64,
                 num_stages: int = 4,
                 output_channels: int = 512,
                 voxel_size: int = 64):
        super(VoxelEncoder, self).__init__()

        self.voxel_size = voxel_size
        self.num_stages = num_stages

        # 输入层
        self.stem = Conv3DBlock(in_channels, base_channels, kernel_size=7,
                                stride=2, padding=3)

        # 多阶段编码器
        self.stages = nn.ModuleList()
        current_channels = base_channels
        current_size = voxel_size // 2  # stem已经下采样2倍

        for i in range(num_stages):
            stage_channels = base_channels * (2 ** i)

            # 下采样层
            if i > 0:
                downsample = Conv3DBlock(current_channels, stage_channels,
                                         kernel_size=3, stride=2, padding=1)
                current_size = current_size // 2
            else:
                downsample = Conv3DBlock(current_channels, stage_channels,
                                         kernel_size=3, stride=1, padding=1)

            # ResNet块
            blocks = [ResNet3DBlock(stage_channels, stage_channels)]
            blocks.append(ResNet3DBlock(stage_channels, stage_channels))

            stage = nn.Sequential(downsample, *blocks)
            self.stages.append(stage)
            current_channels = stage_channels

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # 输出投影
        self.output_proj = nn.Linear(current_channels, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入体素 [B, C, D, H, W]

        Returns:
            global_feature: 全局特征 [B, output_channels]
        """
        # 存储多尺度特征
        multi_scale_features = []

        # Stem
        x = self.stem(x)  # [B, base_channels, D/2, H/2, W/2]
        multi_scale_features.append(x)

        # 多阶段处理
        for stage in self.stages:
            x = stage(x)
            multi_scale_features.append(x)

        # 全局特征
        global_feature = self.global_pool(x)  # [B, C, 1, 1, 1]
        global_feature = global_feature.flatten(1)  # [B, C]
        global_feature = self.output_proj(global_feature)  # [B, output_channels]

        return global_feature


class SparseVoxelEncoder(nn.Module):
    """
    稀疏体素编码器
    专门处理稀疏体素数据，提高内存效率
    """

    def __init__(self,
                 in_channels: int = 3,
                 hidden_channels: int = 128,
                 output_channels: int = 512,
                 num_layers: int = 4):
        super(SparseVoxelEncoder, self).__init__()

        # 点级特征编码器
        self.point_encoder = nn.Sequential(
            nn.Linear(in_channels + 3, hidden_channels),  # +3 for 3D coordinates
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True)
        )

        # 多层感知机
        layers = []
        current_channels = hidden_channels

        for i in range(num_layers - 1):
            next_channels = hidden_channels if i < num_layers - 2 else output_channels
            layers.extend([
                nn.Linear(current_channels, next_channels),
                nn.ReLU(inplace=True) if i < num_layers - 2 else nn.Identity()
            ])
            current_channels = next_channels

        self.mlp = nn.Sequential(*layers)

        # 全局特征聚合
        self.global_aggregator = nn.Sequential(
            nn.Linear(output_channels, output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels, output_channels)
        )

    def forward(self, sparse_voxels: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        处理稀疏体素数据

        Args:
            sparse_voxels: 稀疏体素数据字典或密集体素张量

        Returns:
            global_feature: 全局特征 [B, output_channels]
        """
        # 如果输入是密集体素，转换为稀疏格式
        if isinstance(sparse_voxels, torch.Tensor):
            return self._process_dense_voxels(sparse_voxels)

        # 处理稀疏体素字典
        indices = sparse_voxels['indices']  # [N, 4] (B, X, Y, Z)
        values = sparse_voxels['values']  # [N, C]
        shape = sparse_voxels['shape']  # (B, C, D, H, W)

        batch_size = shape[0]

        if indices.shape[0] == 0:
            # 如果没有有效体素，返回零特征
            return torch.zeros(batch_size, self.global_aggregator[-1].out_features,
                               device=indices.device)

        # 提取3D坐标
        coords_3d = indices[:, 1:4].float()  # [N, 3]

        # 归一化坐标到[-1, 1]
        voxel_size = shape[2]  # 假设D=H=W
        coords_3d = (coords_3d / (voxel_size - 1)) * 2 - 1

        # 组合坐标和颜色特征
        point_features = torch.cat([coords_3d, values], dim=1)  # [N, 3+C]

        # 编码点特征
        encoded_features = self.point_encoder(point_features)  # [N, hidden_channels]
        encoded_features = self.mlp(encoded_features)  # [N, output_channels]

        # 按batch聚合特征
        batch_features = []
        for b in range(batch_size):
            batch_mask = (indices[:, 0] == b)
            if batch_mask.any():
                batch_point_features = encoded_features[batch_mask]
                # 使用平均池化聚合
                aggregated = batch_point_features.mean(dim=0)
            else:
                aggregated = torch.zeros(encoded_features.shape[1],
                                         device=encoded_features.device)

            batch_features.append(aggregated)

        # 堆叠并通过全局聚合器
        global_features = torch.stack(batch_features, dim=0)  # [B, output_channels]
        global_features = self.global_aggregator(global_features)

        return global_features

    def _process_dense_voxels(self, voxels: torch.Tensor) -> torch.Tensor:
        """处理密集体素数据"""
        B, C, D, H, W = voxels.shape
        device = voxels.device

        # 简化处理：全局平均池化 + MLP
        global_features = torch.mean(voxels, dim=[2, 3, 4])  # [B, C]

        # 投影到输出维度
        if hasattr(self, '_dense_proj'):
            projected = self._dense_proj(global_features)
        else:
            # 动态创建投影层
            self._dense_proj = nn.Linear(C, self.global_aggregator[-1].out_features).to(device)
            projected = self._dense_proj(global_features)

        return self.global_aggregator(projected)


class HierarchicalVoxelEncoder(nn.Module):
    """
    层次化体素编码器
    在多个分辨率级别处理体素数据
    """

    def __init__(self,
                 in_channels: int = 3,
                 scales: List[int] = [16, 32, 64],
                 base_channels: int = 64,
                 output_channels: int = 512):
        super(HierarchicalVoxelEncoder, self).__init__()

        self.scales = scales

        # 为每个尺度创建编码器
        self.encoders = nn.ModuleList()
        scale_output_channels = output_channels // len(scales)

        for scale in scales:
            encoder = VoxelEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_stages=3,
                output_channels=scale_output_channels,
                voxel_size=scale
            )
            self.encoders.append(encoder)

        # 特征融合 - 修复维度匹配问题
        total_features = scale_output_channels * len(scales)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, output_channels),
            nn.ReLU(inplace=True),
            nn.Linear(output_channels, output_channels)
        )

    def forward(self, voxels_dict: Union[Dict[int, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        处理多尺度体素数据

        Args:
            voxels_dict: 不同尺度的体素数据字典或单个体素张量

        Returns:
            fused_features: 融合的多尺度特征
        """
        # 如果输入是单个张量，为所有尺度复制
        if isinstance(voxels_dict, torch.Tensor):
            original_voxels = voxels_dict
            voxels_dict = {}
            for scale in self.scales:
                # 重采样到对应尺度
                resampled = F.interpolate(
                    original_voxels,
                    size=(scale, scale, scale),
                    mode='trilinear',
                    align_corners=False
                )
                voxels_dict[scale] = resampled

        scale_features = []

        for i, (encoder, scale) in enumerate(zip(self.encoders, self.scales)):
            if scale in voxels_dict:
                voxels = voxels_dict[scale]
                # 只取全局特征，忽略多尺度特征
                global_feat = encoder(voxels)
                scale_features.append(global_feat)
            else:
                # 如果某个尺度的数据不存在，用零填充
                batch_size = next(iter(voxels_dict.values())).shape[0]
                device = next(iter(voxels_dict.values())).device
                zero_feat = torch.zeros(batch_size, encoder.output_proj.out_features,
                                        device=device)
                scale_features.append(zero_feat)

        # 连接所有尺度的特征
        fused_features = torch.cat(scale_features, dim=1)

        # 特征融合
        fused_features = self.fusion(fused_features)

        return fused_features


class GeometryAwareEncoder(nn.Module):
    """
    几何感知编码器
    显式建模几何结构和空间关系
    """

    def __init__(self,
                 in_channels: int = 3,
                 hidden_channels: int = 128,
                 output_channels: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 4):
        super(GeometryAwareEncoder, self).__init__()

        # 基础特征提取
        self.feature_extractor = VoxelEncoder(
            in_channels=in_channels,
            base_channels=64,
            num_stages=3,
            output_channels=hidden_channels
        )

        # 几何注意力层
        self.geometry_attention = nn.ModuleList([
            GeometryAttentionLayer(hidden_channels, num_heads)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Linear(hidden_channels, output_channels)

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        几何感知编码

        Args:
            voxels: 体素数据 [B, C, D, H, W]

        Returns:
            geometry_features: 几何感知特征
        """
        # 提取基础特征
        features = self.feature_extractor(voxels)

        # 应用几何注意力 (这里简化处理，不使用多尺度特征)
        enhanced_features = features
        for attn_layer in self.geometry_attention:
            enhanced_features = attn_layer(enhanced_features, None)

        # 输出投影
        geometry_features = self.output_proj(enhanced_features)

        return geometry_features


class GeometryAttentionLayer(nn.Module):
    """几何注意力层"""

    def __init__(self, channels: int, num_heads: int = 8):
        super(GeometryAttentionLayer, self).__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        self.out_proj = nn.Linear(channels, channels)

        # 几何位置编码
        self.geometry_pos_encoding = nn.Parameter(torch.randn(1, channels))

    def forward(self, global_features: torch.Tensor,
                spatial_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        几何注意力计算

        Args:
            global_features: 全局特征 [B, C]
            spatial_features: 空间特征 [B, C, D, H, W] (可选)

        Returns:
            enhanced_features: 增强的几何特征
        """
        B, C = global_features.shape

        if spatial_features is not None:
            # 使用空间特征进行注意力计算
            # 展平空间特征
            spatial_flat = spatial_features.flatten(2).transpose(1, 2)  # [B, D*H*W, C]

            # 添加几何位置编码
            spatial_flat = spatial_flat + self.geometry_pos_encoding

            # 计算注意力
            q = self.q_proj(global_features).unsqueeze(1)  # [B, 1, C]
            k = self.k_proj(spatial_flat)  # [B, D*H*W, C]
            v = self.v_proj(spatial_flat)  # [B, D*H*W, C]

            # 多头注意力
            q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

            # 计算注意力权重
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)

            # 应用注意力
            attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, 1, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, C)

            # 输出投影
            enhanced_features = self.out_proj(attn_output.squeeze(1))

            # 残差连接
            enhanced_features = enhanced_features + global_features
        else:
            # 如果没有空间特征，直接使用自注意力
            enhanced_features = self.out_proj(global_features) + global_features

        return enhanced_features


def create_geometry_encoder(config: dict) -> nn.Module:
    """
    工厂函数：根据配置创建几何编码器

    Args:
        config: 配置字典

    Returns:
        encoder: 几何编码器实例
    """
    encoder_type = config.get('type', 'voxel')

    if encoder_type == 'voxel':
        return VoxelEncoder(
            in_channels=config.get('in_channels', 3),
            base_channels=config.get('base_channels', 64),
            num_stages=config.get('num_stages', 4),
            output_channels=config.get('output_channels', 512),
            voxel_size=config.get('voxel_size', 64)
        )
    elif encoder_type == 'sparse':
        return SparseVoxelEncoder(
            in_channels=config.get('in_channels', 3),
            hidden_channels=config.get('hidden_channels', 128),
            output_channels=config.get('output_channels', 512),
            num_layers=config.get('num_layers', 4)
        )
    elif encoder_type == 'hierarchical':
        return HierarchicalVoxelEncoder(
            in_channels=config.get('in_channels', 3),
            scales=config.get('scales', [16, 32, 64]),
            base_channels=config.get('base_channels', 64),
            output_channels=config.get('output_channels', 512)
        )
    elif encoder_type == 'geometry_aware':
        return GeometryAwareEncoder(
            in_channels=config.get('in_channels', 3),
            hidden_channels=config.get('hidden_channels', 128),
            output_channels=config.get('output_channels', 512),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 4)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# 示例使用和测试
if __name__ == "__main__":
    print("=== 测试几何编码器 ===")

    try:
        # 测试基础体素编码器
        print("\n1. 测试基础体素编码器")
        encoder = VoxelEncoder(in_channels=3, voxel_size=64)

        # 创建测试数据
        batch_size = 2
        voxel_input = torch.randn(batch_size, 3, 64, 64, 64)

        # 前向传播
        global_feat, multi_scale_feat = encoder(voxel_input)

        print(f"✅ 输入体素形状: {voxel_input.shape}")
        print(f"✅ 全局特征形状: {global_feat.shape}")
        print(f"✅ 多尺度特征数量: {len(multi_scale_feat)}")
        for i, feat in enumerate(multi_scale_feat):
            print(f"   尺度 {i}: {feat.shape}")

        # 测试稀疏体素编码器
        print("\n2. 测试稀疏体素编码器")
        sparse_encoder = SparseVoxelEncoder(in_channels=3)

        # 创建稀疏体素测试数据
        sparse_data = {
            'indices': torch.tensor([[0, 10, 20, 30], [0, 15, 25, 35],
                                     [1, 5, 15, 25]], dtype=torch.long),
            'values': torch.randn(3, 3),
            'shape': (2, 3, 64, 64, 64)
        }

        sparse_feat = sparse_encoder(sparse_data)
        print(f"✅ 稀疏体素特征形状: {sparse_feat.shape}")

        # 测试层次化体素编码器
        print("\n3. 测试层次化体素编码器")
        hierarchical_encoder = HierarchicalVoxelEncoder()

        # 创建多尺度体素数据
        multi_scale_voxels = {
            16: torch.randn(batch_size, 3, 16, 16, 16),
            32: torch.randn(batch_size, 3, 32, 32, 32),
            64: torch.randn(batch_size, 3, 64, 64, 64)
        }

        hierarchical_feat = hierarchical_encoder(multi_scale_voxels)
        print(f"✅ 层次化特征形状: {hierarchical_feat.shape}")

        # 测试几何感知编码器
        print("\n4. 测试几何感知编码器")
        geo_encoder = GeometryAwareEncoder(in_channels=3)
        geo_feat = geo_encoder(voxel_input)
        print(f"✅ 几何感知特征形状: {geo_feat.shape}")

        print("\n🎉 所有测试通过！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()