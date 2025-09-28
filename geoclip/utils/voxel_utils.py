"""
GeoCLIP - 体素转换工具
将2D深度图转换为3D体素表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import cv2


def depth_to_pointcloud(depth: torch.Tensor,
                       intrinsics: Optional[torch.Tensor] = None,
                       max_depth: float = 10.0) -> List[torch.Tensor]:
    """
    将深度图转换为点云

    Args:
        depth: 深度图 [B, 1, H, W]
        intrinsics: 相机内参 [B, 3, 3] (可选)
        max_depth: 最大深度值

    Returns:
        pointcloud: 点云列表 [B x [N_i, 3]]
    """
    B, C, H, W = depth.shape
    device = depth.device

    # 创建像素坐标网格
    u, v = torch.meshgrid(torch.arange(W, device=device, dtype=torch.float32),
                         torch.arange(H, device=device, dtype=torch.float32),
                         indexing='xy')

    # 展平
    u_flat = u.flatten()  # [H*W]
    v_flat = v.flatten()  # [H*W]
    depth_flat = depth.reshape(B, -1)  # [B, H*W]

    points_list = []

    for b in range(B):
        # 当前批次的内参
        if intrinsics is None:
            # 使用默认内参（假设标准化坐标）
            fx = fy = min(H, W) / 2.0
            cx, cy = W / 2.0, H / 2.0
        else:
            fx = intrinsics[b, 0, 0].item()
            fy = intrinsics[b, 1, 1].item()
            cx = intrinsics[b, 0, 2].item()
            cy = intrinsics[b, 1, 2].item()

        d = depth_flat[b]  # [H*W]

        # 过滤无效深度
        valid_mask = (d > 0) & (d < max_depth) & torch.isfinite(d)

        if valid_mask.sum() == 0:
            # 如果没有有效点，创建一个虚拟点
            points = torch.zeros(1, 3, device=device)
            points_list.append(points)
            continue

        # 获取有效的像素坐标和深度
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        d_valid = d[valid_mask]

        # 计算3D坐标
        x = (u_valid - cx) * d_valid / fx
        y = (v_valid - cy) * d_valid / fy
        z = d_valid

        # 组合3D点
        points = torch.stack([x, y, z], dim=1)  # [N_valid, 3]
        points_list.append(points)

    return points_list


def pointcloud_to_voxel(pointclouds: List[torch.Tensor],
                       colors: Optional[List[torch.Tensor]] = None,
                       voxel_size: int = 64,
                       spatial_range: Tuple[float, float] = (-2.0, 2.0)) -> torch.Tensor:
    """
    将点云转换为体素网格

    Args:
        pointclouds: 点云列表 [B x [N_i, 3]]
        colors: 颜色列表 [B x [N_i, 3]] (可选)
        voxel_size: 体素网格大小
        spatial_range: 空间范围 (min, max)

    Returns:
        voxels: 体素网格 [B, C, D, H, W]
    """
    B = len(pointclouds)
    device = pointclouds[0].device if pointclouds and len(pointclouds[0]) > 0 else torch.device('cpu')

    # 确定通道数
    if colors is not None:
        channels = 4  # RGB + density
    else:
        channels = 1  # density only

    voxels = torch.zeros(B, channels, voxel_size, voxel_size, voxel_size, device=device)

    for b in range(B):
        points = pointclouds[b]  # [N, 3]

        if points.shape[0] == 0:
            continue

        # 标准化到体素网格坐标 [0, voxel_size-1]
        points_norm = (points - spatial_range[0]) / (spatial_range[1] - spatial_range[0])
        points_norm = points_norm * (voxel_size - 1)

        # 转换为整数坐标
        coords = points_norm.round().long()

        # 过滤超出范围的点
        valid_mask = (coords >= 0).all(dim=1) & (coords < voxel_size).all(dim=1)
        coords = coords[valid_mask]

        if coords.shape[0] == 0:
            continue

        # 填充体素
        x_coords, y_coords, z_coords = coords[:, 0], coords[:, 1], coords[:, 2]

        if colors is not None and b < len(colors):
            point_colors = colors[b][valid_mask]  # [N_valid, 3]

            # 密度通道
            voxels[b, 0, x_coords, y_coords, z_coords] = 1.0

            # RGB通道 - 处理多个点映射到同一体素的情况
            for i in range(3):
                # 使用平均值处理重叠
                voxels[b, i+1].index_put_(
                    (x_coords, y_coords, z_coords),
                    point_colors[:, i],
                    accumulate=True
                )
                # 计算每个体素的点数量进行归一化
                count_voxels = torch.zeros_like(voxels[b, 0])
                count_voxels.index_put_(
                    (x_coords, y_coords, z_coords),
                    torch.ones_like(x_coords, dtype=torch.float32),
                    accumulate=True
                )
                # 归一化（避免除零）
                mask = count_voxels > 0
                voxels[b, i+1][mask] = voxels[b, i+1][mask] / count_voxels[mask]
        else:
            # 仅密度
            voxels[b, 0, x_coords, y_coords, z_coords] = 1.0

    return voxels


class DepthToVoxelConverter(nn.Module):
    """
    深度图到体素的转换器
    """

    def __init__(self,
                 voxel_size: int = 64,
                 depth_range: Tuple[float, float] = (0.1, 10.0),
                 spatial_range: Tuple[float, float] = (-2.0, 2.0),
                 use_color: bool = True):
        super(DepthToVoxelConverter, self).__init__()

        self.voxel_size = voxel_size
        self.depth_range = depth_range
        self.spatial_range = spatial_range
        self.use_color = use_color

        # 注册相机内参（可选）
        self.register_buffer('intrinsics', None)

    def set_intrinsics(self, intrinsics: torch.Tensor):
        """设置相机内参"""
        self.register_buffer('intrinsics', intrinsics)

    def images_to_voxels(self, rgbd_images: torch.Tensor) -> torch.Tensor:
        """
        将RGBD图像转换为体素

        Args:
            rgbd_images: RGBD图像 [B, 4, H, W] (RGB + Depth)

        Returns:
            voxels: 体素网格 [B, C, D, H, W]
        """
        B, C, H, W = rgbd_images.shape
        device = rgbd_images.device

        if C != 4:
            raise ValueError(f"期望4通道RGBD图像，但得到{C}通道")

        # 分离RGB和深度
        rgb_images = rgbd_images[:, :3]  # [B, 3, H, W]
        depth_images = rgbd_images[:, 3:4]  # [B, 1, H, W]

        # 转换为点云
        pointclouds = depth_to_pointcloud(
            depth_images,
            self.intrinsics,
            max_depth=self.depth_range[1]
        )

        # 提取对应的颜色
        colors = None
        if self.use_color:
            colors = []
            for b in range(B):
                # 获取有效深度的像素坐标
                depth_b = depth_images[b, 0]  # [H, W]
                rgb_b = rgb_images[b]  # [3, H, W]

                # 创建坐标网格
                u, v = torch.meshgrid(torch.arange(W, device=device, dtype=torch.float32),
                                     torch.arange(H, device=device, dtype=torch.float32),
                                     indexing='xy')

                # 展平并过滤有效深度
                u_flat = u.flatten()
                v_flat = v.flatten()
                depth_flat = depth_b.flatten()

                valid_mask = (depth_flat > 0) & (depth_flat < self.depth_range[1]) & torch.isfinite(depth_flat)

                if valid_mask.sum() > 0:
                    u_valid = u_flat[valid_mask].long()
                    v_valid = v_flat[valid_mask].long()

                    # 确保坐标在范围内
                    u_valid = torch.clamp(u_valid, 0, W-1)
                    v_valid = torch.clamp(v_valid, 0, H-1)

                    # 提取对应位置的RGB值
                    rgb_values = rgb_b[:, v_valid, u_valid].t()  # [N_valid, 3]
                    colors.append(rgb_values)
                else:
                    colors.append(torch.zeros(1, 3, device=device))

        # 转换为体素
        voxels = pointcloud_to_voxel(
            pointclouds,
            colors,
            self.voxel_size,
            self.spatial_range
        )

        return voxels

    def depth_to_voxel(self, depth_images: torch.Tensor) -> torch.Tensor:
        """
        仅从深度图生成体素（无颜色信息）

        Args:
            depth_images: 深度图 [B, 1, H, W]

        Returns:
            voxels: 体素网格 [B, 1, D, H, W]
        """
        # 转换为点云
        pointclouds = depth_to_pointcloud(
            depth_images,
            self.intrinsics,
            max_depth=self.depth_range[1]
        )

        # 转换为体素（无颜色）
        voxels = pointcloud_to_voxel(
            pointclouds,
            colors=None,
            voxel_size=self.voxel_size,
            spatial_range=self.spatial_range
        )

        return voxels


def voxel_grid_sampling(voxels: torch.Tensor,
                       target_size: int,
                       mode: str = 'trilinear') -> torch.Tensor:
    """
    体素网格重采样

    Args:
        voxels: 输入体素 [B, C, D, H, W]
        target_size: 目标尺寸
        mode: 插值模式

    Returns:
        resampled_voxels: 重采样体素 [B, C, target_size, target_size, target_size]
    """
    return F.interpolate(voxels, size=(target_size, target_size, target_size),
                        mode=mode, align_corners=False)


def voxel_to_mesh(voxels: torch.Tensor, threshold: float = 0.5):
    """
    将体素转换为网格（使用marching cubes算法）
    注意：这需要额外的库如scikit-image

    Args:
        voxels: 体素数据 [D, H, W]
        threshold: 等值面阈值

    Returns:
        vertices, faces: 网格顶点和面
    """
    try:
        from skimage import measure

        # 转换为numpy
        voxel_np = voxels.cpu().numpy()

        # Marching cubes
        vertices, faces, _, _ = measure.marching_cubes(voxel_np, threshold)

        return vertices, faces
    except ImportError:
        raise ImportError("需要安装 scikit-image 来使用网格转换功能")


def create_multiscale_voxels(rgbd_images: torch.Tensor,
                            scales: List[int] = [16, 32, 64],
                            depth_range: Tuple[float, float] = (0.1, 10.0)) -> Dict[int, torch.Tensor]:
    """
    创建多尺度体素表示

    Args:
        rgbd_images: RGBD图像 [B, 4, H, W]
        scales: 体素尺度列表
        depth_range: 深度范围

    Returns:
        multiscale_voxels: 多尺度体素字典 {scale: voxels}
    """
    multiscale_voxels = {}

    for scale in scales:
        converter = DepthToVoxelConverter(
            voxel_size=scale,
            depth_range=depth_range,
            use_color=True
        )

        voxels = converter.images_to_voxels(rgbd_images)
        multiscale_voxels[scale] = voxels

    return multiscale_voxels


# 测试和使用示例
def test_voxel_conversion():
    """测试体素转换功能"""
    print("=== 测试体素转换工具 ===")

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 1. 创建测试数据
        print("\n1. 创建测试数据")
        batch_size = 2
        H, W = 128, 128

        # 创建RGBD图像
        rgb_images = torch.randn(batch_size, 3, H, W, device=device)

        # 创建合成深度图（中心近，边缘远）
        y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        depth_pattern = torch.exp(-(x**2 + y**2))  # 高斯分布
        depth_images = depth_pattern.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        depth_images = depth_images.to(device) * 5.0 + 1.0  # 范围 [1, 6]

        rgbd_images = torch.cat([rgb_images, depth_images], dim=1)

        print(f"✅ RGBD图像形状: {rgbd_images.shape}")
        print(f"   深度范围: {depth_images.min():.3f} - {depth_images.max():.3f}")

        # 2. 测试单尺度转换
        print("\n2. 测试单尺度体素转换")
        converter = DepthToVoxelConverter(
            voxel_size=64,
            depth_range=(0.5, 10.0),
            use_color=True
        )

        voxels = converter.images_to_voxels(rgbd_images)
        print(f"✅ 体素形状: {voxels.shape}")
        print(f"   体素占用率: {(voxels[:, 0] > 0).float().mean():.3f}")

        # 3. 测试多尺度转换
        print("\n3. 测试多尺度体素转换")
        multiscale_voxels = create_multiscale_voxels(rgbd_images, scales=[16, 32, 64])

        for scale, voxels_scale in multiscale_voxels.items():
            occupancy = (voxels_scale[:, 0] > 0).float().mean()
            print(f"✅ 尺度 {scale}: {voxels_scale.shape}, 占用率: {occupancy:.3f}")

        # 4. 测试体素重采样
        print("\n4. 测试体素重采样")
        original_voxels = multiscale_voxels[64]
        resampled_voxels = voxel_grid_sampling(original_voxels, target_size=32)
        print(f"✅ 重采样: {original_voxels.shape} -> {resampled_voxels.shape}")

        # 5. 测试仅深度转换
        print("\n5. 测试仅深度体素转换")
        depth_only_voxels = converter.depth_to_voxel(depth_images)
        print(f"✅ 深度体素形状: {depth_only_voxels.shape}")

        print("\n🎉 体素转换测试完成!")

        return multiscale_voxels

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_voxel_conversion()