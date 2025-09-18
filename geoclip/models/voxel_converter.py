"""
GeoCLIP - 体素转换模块
将深度图和RGB图像转换为3D体素表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import open3d as o3d


class VoxelConverter(nn.Module):
    """
    深度图像到体素网格的转换器
    """

    def __init__(self,
                 voxel_size: int = 64,
                 world_size: float = 2.0,
                 min_depth: float = 0.1,
                 max_depth: float = 10.0,
                 device: str = "cuda"):
        super(VoxelConverter, self).__init__()

        self.voxel_size = voxel_size
        self.world_size = world_size  # 3D空间的物理尺寸
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = device

        # 体素网格的分辨率
        self.voxel_resolution = world_size / voxel_size

        # 预计算体素网格坐标
        self.register_buffer('voxel_coords', self._create_voxel_coordinates())

    def _create_voxel_coordinates(self):
        """创建体素网格的3D坐标"""
        # 创建规则网格
        x = torch.linspace(-self.world_size/2, self.world_size/2, self.voxel_size)
        y = torch.linspace(-self.world_size/2, self.world_size/2, self.voxel_size)
        z = torch.linspace(-self.world_size/2, self.world_size/2, self.voxel_size)

        # 创建网格坐标
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        # 展平并组合 [N, 3]
        coords = torch.stack([grid_x.flatten(),
                             grid_y.flatten(),
                             grid_z.flatten()], dim=1)

        return coords

    def depth_to_pointcloud(self,
                           rgb: torch.Tensor,
                           depth: torch.Tensor,
                           camera_intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        将深度图转换为点云

        Args:
            rgb: RGB图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]
            camera_intrinsics: 相机内参 [B, 3, 3] (可选)

        Returns:
            pointcloud: 点云 [B, N, 6] (XYZ + RGB)
        """
        B, _, H, W = rgb.shape

        # 默认相机内参
        if camera_intrinsics is None:
            fx = fy = W / 2.0  # 简单假设
            cx, cy = W / 2.0, H / 2.0
            camera_intrinsics = torch.tensor([[[fx, 0, cx],
                                             [0, fy, cy],
                                             [0, 0, 1]]],
                                           dtype=rgb.dtype, device=rgb.device)
            camera_intrinsics = camera_intrinsics.repeat(B, 1, 1)

        # 创建像素坐标网格
        u, v = torch.meshgrid(torch.arange(W, device=rgb.device, dtype=torch.float32),
                             torch.arange(H, device=rgb.device, dtype=torch.float32),
                             indexing='ij')
        u = u.T  # [H, W]
        v = v.T  # [H, W]

        pointclouds = []

        for b in range(B):
            # 提取单个样本的数据
            rgb_b = rgb[b].permute(1, 2, 0)  # [H, W, 3]
            depth_b = depth[b, 0]  # [H, W]
            K = camera_intrinsics[b]  # [3, 3]

            # 过滤有效深度值
            valid_mask = (depth_b > self.min_depth) & (depth_b < self.max_depth)

            if not valid_mask.any():
                # 如果没有有效深度，返回空点云
                empty_pc = torch.zeros(0, 6, device=rgb.device)
                pointclouds.append(empty_pc)
                continue

            # 提取有效的像素坐标和深度
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            depth_valid = depth_b[valid_mask]
            rgb_valid = rgb_b[valid_mask]  # [N, 3]

            # 转换到相机坐标系
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            x = (u_valid - cx) * depth_valid / fx
            y = (v_valid - cy) * depth_valid / fy
            z = depth_valid

            # 组合3D坐标和颜色
            xyz = torch.stack([x, y, z], dim=1)  # [N, 3]
            pointcloud = torch.cat([xyz, rgb_valid], dim=1)  # [N, 6]

            pointclouds.append(pointcloud)

        return pointclouds

    def pointcloud_to_voxel(self,
                           pointclouds: List[torch.Tensor],
                           features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        将点云转换为体素表示

        Args:
            pointclouds: 点云列表，每个元素为 [N, 6] (XYZ + RGB)
            features: 额外特征 [B, C, H, W] (可选)

        Returns:
            voxels: 体素网格 [B, C, D, H, W]
        """
        B = len(pointclouds)

        if features is not None:
            C = features.shape[1] + 3  # RGB + 额外特征
        else:
            C = 3  # 仅RGB

        voxels = torch.zeros(B, C, self.voxel_size, self.voxel_size, self.voxel_size,
                           device=self.device, dtype=torch.float32)

        for b, pc in enumerate(pointclouds):
            if pc.shape[0] == 0:  # 空点云
                continue

            xyz = pc[:, :3]  # [N, 3]
            rgb = pc[:, 3:6]  # [N, 3]

            # 将3D坐标映射到体素索引
            voxel_indices = self._world_to_voxel_indices(xyz)

            # 过滤有效的体素索引
            valid_mask = (voxel_indices >= 0).all(dim=1) & \
                        (voxel_indices < self.voxel_size).all(dim=1)

            if not valid_mask.any():
                continue

            voxel_indices = voxel_indices[valid_mask]
            rgb = rgb[valid_mask]

            # 填充体素网格
            for i in range(voxel_indices.shape[0]):
                x, y, z = voxel_indices[i]
                voxels[b, :3, x, y, z] = rgb[i]  # RGB通道

                # 如果有额外特征，使用双线性插值从特征图中采样
                if features is not None:
                    # 将3D坐标投影回2D图像空间
                    # 这里需要相机参数，简化处理
                    feat_value = self._sample_feature_at_3d_point(
                        features[b], xyz[valid_mask][i:i+1]
                    )
                    voxels[b, 3:, x, y, z] = feat_value

        return voxels

    def _world_to_voxel_indices(self, xyz: torch.Tensor) -> torch.Tensor:
        """将世界坐标转换为体素索引"""
        # 归一化到[-1, 1]
        normalized = xyz / (self.world_size / 2)

        # 转换到体素索引[0, voxel_size-1]
        indices = ((normalized + 1) / 2 * self.voxel_size).long()

        return indices

    def _sample_feature_at_3d_point(self,
                                   features: torch.Tensor,
                                   xyz: torch.Tensor) -> torch.Tensor:
        """在3D点处采样2D特征"""
        # 简化：假设正交投影
        # 实际应用中需要使用正确的相机投影
        C, H, W = features.shape

        # 将3D点投影到2D (简化版本)
        x_2d = (xyz[:, 0] + self.world_size/2) / self.world_size * W
        y_2d = (xyz[:, 1] + self.world_size/2) / self.world_size * H

        # 归一化到[-1, 1]用于grid_sample
        x_norm = (x_2d / W) * 2 - 1
        y_norm = (y_2d / H) * 2 - 1

        # 创建采样网格
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).unsqueeze(0)

        # 双线性插值采样
        sampled = F.grid_sample(features.unsqueeze(0), grid,
                               mode='bilinear', padding_mode='zeros',
                               align_corners=False)

        return sampled[0, :, 0, 0]  # [C]

    def forward(self,
                rgb: torch.Tensor,
                depth: torch.Tensor,
                features: Optional[torch.Tensor] = None,
                camera_intrinsics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        完整的RGB-D到体素转换流程

        Args:
            rgb: RGB图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]
            features: 额外特征 [B, C, H, W] (可选)
            camera_intrinsics: 相机内参 [B, 3, 3] (可选)

        Returns:
            voxels: 体素网格 [B, C, D, H, W]
        """
        # Step 1: 深度图转点云
        pointclouds = self.depth_to_pointcloud(rgb, depth, camera_intrinsics)

        # Step 2: 点云转体素
        voxels = self.pointcloud_to_voxel(pointclouds, features)

        return voxels


class SparseVoxelConverter(nn.Module):
    """
    稀疏体素转换器 - 内存高效版本
    """

    def __init__(self,
                 voxel_size: int = 64,
                 world_size: float = 2.0,
                 min_depth: float = 0.1,
                 max_depth: float = 10.0,
                 device: str = "cuda"):
        super(SparseVoxelConverter, self).__init__()

        self.voxel_size = voxel_size
        self.world_size = world_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = device

    def forward(self,
                rgb: torch.Tensor,
                depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        返回稀疏体素表示

        Returns:
            dict包含:
            - indices: 有效体素的索引 [N, 4] (B, X, Y, Z)
            - values: 有效体素的值 [N, C]
            - shape: 完整体素网格的形状
        """
        B, _, H, W = rgb.shape

        sparse_voxels = {
            'indices': [],
            'values': [],
            'shape': (B, 3, self.voxel_size, self.voxel_size, self.voxel_size)
        }

        # 处理每个batch
        for b in range(B):
            # 转换为点云
            pc = self._depth_to_pointcloud_single(rgb[b], depth[b])

            if pc.shape[0] > 0:
                # 转换为体素索引
                voxel_indices = self._world_to_voxel_indices(pc[:, :3])

                # 过滤有效索引
                valid_mask = (voxel_indices >= 0).all(dim=1) & \
                           (voxel_indices < self.voxel_size).all(dim=1)

                if valid_mask.any():
                    valid_indices = voxel_indices[valid_mask]
                    valid_colors = pc[valid_mask, 3:6]

                    # 添加batch维度
                    batch_indices = torch.full((valid_indices.shape[0], 1), b,
                                             device=self.device, dtype=torch.long)
                    full_indices = torch.cat([batch_indices, valid_indices], dim=1)

                    sparse_voxels['indices'].append(full_indices)
                    sparse_voxels['values'].append(valid_colors)

        # 合并所有batch的结果
        if sparse_voxels['indices']:
            sparse_voxels['indices'] = torch.cat(sparse_voxels['indices'], dim=0)
            sparse_voxels['values'] = torch.cat(sparse_voxels['values'], dim=0)
        else:
            # 如果没有有效体素，返回空tensor
            sparse_voxels['indices'] = torch.empty(0, 4, dtype=torch.long, device=self.device)
            sparse_voxels['values'] = torch.empty(0, 3, dtype=torch.float32, device=self.device)

        return sparse_voxels

    def _depth_to_pointcloud_single(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """单个样本的深度图转点云"""
        C, H, W = rgb.shape

        # 创建像素坐标
        u, v = torch.meshgrid(torch.arange(W, device=rgb.device, dtype=torch.float32),
                             torch.arange(H, device=rgb.device, dtype=torch.float32),
                             indexing='ij')
        u = u.T.flatten()
        v = v.T.flatten()

        # 展平深度和RGB
        depth_flat = depth[0].flatten()  # [H*W]
        rgb_flat = rgb.permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]

        # 过滤有效深度
        valid_mask = (depth_flat > self.min_depth) & (depth_flat < self.max_depth)

        if not valid_mask.any():
            return torch.empty(0, 6, device=rgb.device)

        # 提取有效数据
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_flat[valid_mask]
        rgb_valid = rgb_flat[valid_mask]

        # 简化相机模型 (假设焦距为图像宽度的一半)
        fx = fy = W / 2.0
        cx, cy = W / 2.0, H / 2.0

        # 转换为3D坐标
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid

        # 组合点云
        xyz = torch.stack([x, y, z], dim=1)
        pointcloud = torch.cat([xyz, rgb_valid], dim=1)

        return pointcloud

    def _world_to_voxel_indices(self, xyz: torch.Tensor) -> torch.Tensor:
        """将世界坐标转换为体素索引"""
        # 归一化到[-1, 1]
        normalized = xyz / (self.world_size / 2)

        # 转换到体素索引[0, voxel_size-1]
        indices = ((normalized + 1) / 2 * self.voxel_size).long()

        # 确保索引在有效范围内
        indices = torch.clamp(indices, 0, self.voxel_size - 1)

        return indices


class AdaptiveVoxelConverter(nn.Module):
    """
    自适应体素转换器
    根据场景深度范围动态调整体素空间
    """

    def __init__(self,
                 voxel_size: int = 64,
                 min_depth: float = 0.1,
                 device: str = "cuda"):
        super(AdaptiveVoxelConverter, self).__init__()

        self.voxel_size = voxel_size
        self.min_depth = min_depth
        self.device = device

    def forward(self,
                rgb: torch.Tensor,
                depth: torch.Tensor) -> torch.Tensor:
        """
        自适应体素转换

        Args:
            rgb: RGB图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]

        Returns:
            voxels: 体素网格 [B, 3, D, H, W]
        """
        B = rgb.shape[0]
        voxels_list = []

        for b in range(B):
            # 计算每个样本的深度范围
            depth_b = depth[b, 0]
            valid_mask = depth_b > self.min_depth

            if not valid_mask.any():
                # 如果没有有效深度，创建空体素网格
                empty_voxels = torch.zeros(3, self.voxel_size, self.voxel_size,
                                         self.voxel_size, device=self.device)
                voxels_list.append(empty_voxels)
                continue

            valid_depths = depth_b[valid_mask]
            min_d, max_d = valid_depths.min(), valid_depths.max()

            # 自适应世界空间大小
            world_size = float(max_d - min_d + 0.5)  # 添加边界

            # 创建该样本的转换器
            converter = VoxelConverter(
                voxel_size=self.voxel_size,
                world_size=world_size,
                min_depth=float(min_d - 0.1),
                max_depth=float(max_d + 0.1),
                device=self.device
            )

            # 转换单个样本
            single_voxel = converter(rgb[b:b+1], depth[b:b+1])
            voxels_list.append(single_voxel[0])

        # 堆叠所有样本
        voxels = torch.stack(voxels_list, dim=0)

        return voxels


def create_voxel_converter(config: dict) -> nn.Module:
    """
    工厂函数：根据配置创建体素转换器

    Args:
        config: 配置字典

    Returns:
        voxel_converter: 体素转换器实例
    """
    converter_type = config.get('type', 'dense')

    common_params = {
        'voxel_size': config.get('voxel_size', 64),
        'world_size': config.get('world_size', 2.0),
        'min_depth': config.get('min_depth', 0.1),
        'max_depth': config.get('max_depth', 10.0),
        'device': config.get('device', 'cuda')
    }

    if converter_type == 'dense':
        return VoxelConverter(**common_params)
    elif converter_type == 'sparse':
        return SparseVoxelConverter(**{k: v for k, v in common_params.items()
                                     if k != 'world_size'})
    elif converter_type == 'adaptive':
        return AdaptiveVoxelConverter(
            voxel_size=common_params['voxel_size'],
            min_depth=common_params['min_depth'],
            device=common_params['device']
        )
    else:
        raise ValueError(f"Unknown converter type: {converter_type}")


# 工具函数
def visualize_voxel_grid(voxels: torch.Tensor,
                        threshold: float = 0.1,
                        save_path: Optional[str] = None):
    """
    可视化体素网格

    Args:
        voxels: 体素网格 [C, D, H, W] 或 [B, C, D, H, W]
        threshold: 显示阈值
        save_path: 保存路径
    """
    if voxels.dim() == 5:
        voxels = voxels[0]  # 取第一个batch

    # 转换为numpy
    voxels_np = voxels.detach().cpu().numpy()

    if voxels_np.shape[0] >= 3:  # RGB通道
        # 计算体素的占用情况
        occupancy = np.linalg.norm(voxels_np[:3], axis=0) > threshold
        colors = voxels_np[:3].transpose(1, 2, 3, 0)  # [D, H, W, 3]
    else:
        occupancy = voxels_np[0] > threshold
        colors = None

    # 获取非空体素的坐标
    occupied_coords = np.where(occupancy)

    if len(occupied_coords[0]) == 0:
        print("没有找到非空体素")
        return

    # 创建点云
    points = np.column_stack(occupied_coords)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    # 添加颜色
    if colors is not None:
        point_colors = colors[occupied_coords]
        pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # 可视化
    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
        print(f"体素可视化已保存到: {save_path}")
    else:
        o3d.visualization.draw_geometries([pcd])


# 示例使用
if __name__ == "__main__":
    # 创建体素转换器
    converter = VoxelConverter(voxel_size=32, world_size=2.0)

    # 测试数据
    batch_size = 2
    rgb = torch.randn(batch_size, 3, 256, 256).cuda()
    depth = torch.rand(batch_size, 1, 256, 256).cuda() * 5 + 0.5  # 0.5-5.5深度范围

    # 转换为体素
    voxels = converter(rgb, depth)

    print(f"输入RGB形状: {rgb.shape}")
    print(f"输入深度形状: {depth.shape}")
    print(f"输出体素形状: {voxels.shape}")
    print(f"非零体素数量: {(voxels > 0).sum().item()}")

    # 测试稀疏转换器
    sparse_converter = SparseVoxelConverter(voxel_size=32)
    sparse_result = sparse_converter(rgb, depth)

    print(f"稀疏体素索引形状: {sparse_result['indices'].shape}")
    print(f"稀疏体素值形状: {sparse_result['values'].shape}")