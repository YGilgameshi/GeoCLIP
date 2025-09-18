import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Dict, Any, List, Optional, Tuple
import cv2
from pathlib import Path


class GeoCLIPMetrics:
    """GeoCLIP评估指标"""

    @staticmethod
    def calculate_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """计算AUROC"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_scores)

    @staticmethod
    def calculate_ap(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """计算Average Precision"""
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_true, y_scores)

    @staticmethod
    def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算F1 Score"""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred)

    @staticmethod
    def calculate_pixel_auroc(gt_masks: List[np.ndarray],
                              pred_maps: List[np.ndarray]) -> float:
        """计算像素级AUROC"""
        # 展平所有mask和预测图
        gt_flat = np.concatenate([mask.flatten() for mask in gt_masks])
        pred_flat = np.concatenate([pred.flatten() for pred in pred_maps])

        return GeoCLIPMetrics.calculate_auroc(gt_flat, pred_flat)

    @staticmethod
    def calculate_image_auroc(gt_labels: List[int],
                              pred_scores: List[float]) -> float:
        """计算图像级AUROC"""
        return GeoCLIPMetrics.calculate_auroc(np.array(gt_labels),
                                              np.array(pred_scores))