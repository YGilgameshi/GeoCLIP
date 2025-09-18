import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Dict, Any, List, Optional, Tuple
import cv2
from pathlib import Path


class GeoCLIPVisualizer:
    """GeoCLIP可视化工具"""

    @staticmethod
    def visualize_depth_comparison(rgb: np.ndarray,
                                   depth_gt: Optional[np.ndarray],
                                   depth_pred: np.ndarray,
                                   save_path: Optional[str] = None):
        """可视化深度估计结果对比"""
        fig, axes = plt.subplots(1, 3 if depth_gt is not None else 2,
                                 figsize=(15, 5))

        if depth_gt is None:
            axes = [axes[0], axes[1]]

        # RGB图像
        axes[0].imshow(rgb)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')

        # 预测深度
        im1 = axes[1].imshow(depth_pred, cmap='viridis')
        axes[1].set_title('Predicted Depth')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        # 真实深度（如果有）
        if depth_gt is not None:
            im2 = axes[2].imshow(depth_gt, cmap='viridis')
            axes[2].set_title('Ground Truth Depth')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    @staticmethod
    def visualize_voxel_grid(voxels: torch.Tensor,
                             threshold: float = 0.1,
                             max_points: int = 10000,
                             save_path: Optional[str] = None):
        """可视化体素网格"""
        if voxels.dim() == 5:
            voxels = voxels[0]  # 取第一个batch

        # 转换为numpy
        voxels_np = voxels.detach().cpu().numpy()

        # 计算占用情况
        if voxels_np.shape[0] >= 3:  # RGB通道
            occupancy = np.linalg.norm(voxels_np[:3], axis=0) > threshold
            colors = voxels_np[:3].transpose(1, 2, 3, 0)
        else:
            occupancy = voxels_np[0] > threshold
            colors = None

        # 获取非空体素坐标
        occupied_coords = np.where(occupancy)

        if len(occupied_coords[0]) == 0:
            print("没有找到非空体素")
            return

        # 限制点数量以提高可视化性能
        num_points = len(occupied_coords[0])
        if num_points > max_points:
            indices = np.random.choice(num_points, max_points, replace=False)
            occupied_coords = (occupied_coords[0][indices],
                               occupied_coords[1][indices],
                               occupied_coords[2][indices])

        # 创建点云
        points = np.column_stack(occupied_coords).astype(np.float64)

        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 添加颜色
        if colors is not None:
            point_colors = colors[occupied_coords]
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
        else:
            # 使用默认蓝色
            pcd.paint_uniform_color([0.3, 0.6, 1.0])

        # 可视化或保存
        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"体素可视化已保存到: {save_path}")
        else:
            # 创建可视化窗口
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Voxel Visualization")
            vis.add_geometry(pcd)

            # 设置视角
            view_control = vis.get_view_control()
            view_control.set_lookat([32, 32, 32])
            view_control.set_up([0, 0, 1])
            view_control.set_front([1, 0, 0])

            vis.run()
            vis.destroy_window()

    @staticmethod
    def visualize_anomaly_detection_result(rgb: np.ndarray,
                                           depth: np.ndarray,
                                           anomaly_map_2d: np.ndarray,
                                           anomaly_score: float,
                                           save_path: Optional[str] = None):
        """可视化异常检测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RGB图像
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB Image')
        axes[0, 0].axis('off')

        # 深度图
        im1 = axes[0, 1].imshow(depth, cmap='viridis')
        axes[0, 1].set_title('Depth Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        # 异常热力图
        im2 = axes[1, 0].imshow(anomaly_map_2d, cmap='hot')
        axes[1, 0].set_title('Anomaly Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0])

        # 叠加结果
        overlay = rgb.copy()
        anomaly_normalized = (anomaly_map_2d - anomaly_map_2d.min()) / \
                             (anomaly_map_2d.max() - anomaly_map_2d.min())

        # 将异常区域用红色高亮
        red_mask = anomaly_normalized > 0.5
        overlay[red_mask] = overlay[red_mask] * 0.5 + np.array([255, 0, 0]) * 0.5

        axes[1, 1].imshow(overlay.astype(np.uint8))
        axes[1, 1].set_title(f'Overlay (Score: {anomaly_score:.3f})')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()
