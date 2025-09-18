import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import cv2


class GeoCLIPTransforms:
    """GeoCLIP专用的数据变换"""

    @staticmethod
    def normalize_depth(depth: torch.Tensor,
                        min_depth: float = 0.1,
                        max_depth: float = 10.0) -> torch.Tensor:
        """深度图归一化"""
        depth = torch.clamp(depth, min_depth, max_depth)
        depth = (depth - min_depth) / (max_depth - min_depth)
        return depth

    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray,
                                 target_size: Tuple[int, int],
                                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """保持宽高比的图像缩放"""
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # 计算缩放比例
        scale = min(target_w / w, target_h / h)

        # 新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # 创建目标尺寸的图像并居中放置
        result = np.zeros((target_h, target_w, image.shape[2]) if len(image.shape) == 3
                          else (target_h, target_w), dtype=image.dtype)

        start_h = (target_h - new_h) // 2
        start_w = (target_w - new_w) // 2

        if len(image.shape) == 3:
            result[start_h:start_h + new_h, start_w:start_w + new_w, :] = resized
        else:
            result[start_h:start_h + new_h, start_w:start_w + new_w] = resized

        return result

    @staticmethod
    def create_camera_intrinsics(image_size: Tuple[int, int],
                                 fov: float = 60.0) -> np.ndarray:
        """根据图像尺寸和视场角创建相机内参"""
        h, w = image_size

        # 计算焦距
        focal_length = w / (2 * np.tan(np.radians(fov / 2)))

        # 相机内参矩阵
        K = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)

        return K

    @staticmethod
    def augment_depth(depth: torch.Tensor,
                      noise_std: float = 0.01,
                      dropout_prob: float = 0.1) -> torch.Tensor:
        """深度图增强"""
        if torch.rand(1) < 0.5:  # 50%概率应用增强
            # 添加高斯噪声
            noise = torch.randn_like(depth) * noise_std
            depth = depth + noise

            # 随机dropout
            if dropout_prob > 0:
                dropout_mask = torch.rand_like(depth) > dropout_prob
                depth = depth * dropout_mask.float()

        return depth
