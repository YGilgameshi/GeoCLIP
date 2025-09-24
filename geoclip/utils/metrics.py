"""
GeoCLIP - 异常检测指标
计算各种异常检测性能指标
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, f1_score, precision_score, recall_score
)
import warnings

warnings.filterwarnings("ignore")


class AnomalyMetrics:
    """
    异常检测指标计算器
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """重置累积指标"""
        self.all_predictions = []
        self.all_labels = []
        self.all_scores = []

    def update(self, predictions: Union[np.ndarray, torch.Tensor],
               labels: Union[np.ndarray, torch.Tensor],
               scores: Optional[Union[np.ndarray, torch.Tensor]] = None):
        """
        更新累积指标

        Args:
            predictions: 预测结果 [N]
            labels: 真实标签 [N]
            scores: 置信度分数 [N] (可选)
        """
        # 转换为numpy
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if scores is not None and torch.is_tensor(scores):
            scores = scores.cpu().numpy()

        self.all_predictions.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())

        if scores is not None:
            self.all_scores.extend(scores.flatten())
        else:
            self.all_scores.extend(predictions.flatten())

    def compute_metrics(self, predictions: Optional[Union[np.ndarray, torch.Tensor]] = None,
                        labels: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            predictions: 预测结果 [N] (可选，如果不提供则使用累积的)
            labels: 真实标签 [N] (可选，如果不提供则使用累积的)

        Returns:
            metrics: 指标字典
        """
        # 如果提供了新的数据，则使用新数据
        if predictions is not None and labels is not None:
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()

            pred_array = predictions.flatten()
            label_array = labels.flatten()
            score_array = pred_array.copy()
        else:
            # 使用累积的数据
            if not self.all_predictions or not self.all_labels:
                return {}

            pred_array = np.array(self.all_predictions)
            label_array = np.array(self.all_labels)
            score_array = np.array(self.all_scores)

        metrics = {}

        try:
            # 基础分类指标
            if len(np.unique(label_array)) > 1:  # 确保有正负样本

                # AUC-ROC
                metrics['auc'] = roc_auc_score(label_array, score_array)

                # AUC-PR
                metrics['ap'] = average_precision_score(label_array, score_array)

                # 获取最优阈值
                optimal_threshold = self._get_optimal_threshold(label_array, score_array)
                metrics['optimal_threshold'] = optimal_threshold

                # 基于最优阈值的二分类指标
                pred_binary = (score_array >= optimal_threshold).astype(int)

                metrics['f1'] = f1_score(label_array, pred_binary)
                metrics['precision'] = precision_score(label_array, pred_binary)
                metrics['recall'] = recall_score(label_array, pred_binary)

                # 混淆矩阵相关
                tn, fp, fn, tp = confusion_matrix(label_array, pred_binary).ravel()

                metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

                # 假阳性率和假阴性率
                metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

                # 平衡准确率
                metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2

            else:
                # 如果只有一个类别，返回默认值
                metrics.update({
                    'auc': 0.5, 'ap': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                    'accuracy': 0.0, 'specificity': 0.0, 'sensitivity': 0.0,
                    'fpr': 0.0, 'fnr': 0.0, 'balanced_accuracy': 0.0,
                    'optimal_threshold': self.threshold
                })

        except Exception as e:
            print(f"计算指标时出错: {e}")
            # 返回默认指标
            metrics.update({
                'auc': 0.0, 'ap': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0,
                'accuracy': 0.0, 'specificity': 0.0, 'sensitivity': 0.0,
                'fpr': 0.0, 'fnr': 0.0, 'balanced_accuracy': 0.0,
                'optimal_threshold': self.threshold
            })

        return metrics

    def _get_optimal_threshold(self, labels: np.ndarray, scores: np.ndarray) -> float:
        """
        计算最优阈值（基于F1分数）

        Args:
            labels: 真实标签
            scores: 预测分数

        Returns:
            optimal_threshold: 最优阈值
        """
        try:
            precision, recall, thresholds = precision_recall_curve(labels, scores)

            # 计算F1分数
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

            # 找到最优阈值
            optimal_idx = np.argmax(f1_scores)

            if optimal_idx < len(thresholds):
                return thresholds[optimal_idx]
            else:
                return self.threshold

        except:
            return self.threshold

    def get_roc_curve_data(self, predictions: Optional[np.ndarray] = None,
                           labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取ROC曲线数据

        Returns:
            fpr, tpr, thresholds: ROC曲线数据
        """
        if predictions is not None and labels is not None:
            score_array = predictions
            label_array = labels
        else:
            score_array = np.array(self.all_scores)
            label_array = np.array(self.all_labels)

        try:
            fpr, tpr, thresholds = roc_curve(label_array, score_array)
            return fpr, tpr, thresholds
        except:
            return np.array([0, 1]), np.array([0, 1]), np.array([0, 1])

    def get_pr_curve_data(self, predictions: Optional[np.ndarray] = None,
                          labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取Precision-Recall曲线数据

        Returns:
            precision, recall, thresholds: PR曲线数据
        """
        if predictions is not None and labels is not None:
            score_array = predictions
            label_array = labels
        else:
            score_array = np.array(self.all_scores)
            label_array = np.array(self.all_labels)

        try:
            precision, recall, thresholds = precision_recall_curve(label_array, score_array)
            return precision, recall, thresholds
        except:
            return np.array([1, 0]), np.array([0, 1]), np.array([0, 1])

    def compute_class_metrics(self, predictions: np.ndarray,
                              labels: np.ndarray,
                              class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        计算每个类别的指标

        Args:
            predictions: 预测结果 [N]
            labels: 真实标签 [N]
            class_names: 类别名称列表

        Returns:
            class_metrics: 每个类别的指标字典
        """
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        unique_classes = np.unique(labels)

        if class_names is None:
            class_names = [f'class_{i}' for i in unique_classes]

        class_metrics = {}

        for i, class_id in enumerate(unique_classes):
            if i < len(class_names):
                class_name = class_names[i]
            else:
                class_name = f'class_{class_id}'

            # 创建二分类标签（当前类 vs 其他）
            binary_labels = (labels == class_id).astype(int)
            binary_preds = (predictions == class_id).astype(int)

            # 计算指标
            class_metrics[class_name] = {
                'precision': precision_score(binary_labels, binary_preds, zero_division=0),
                'recall': recall_score(binary_labels, binary_preds, zero_division=0),
                'f1': f1_score(binary_labels, binary_preds, zero_division=0),
                'support': np.sum(binary_labels)
            }

        return class_metrics

    def print_metrics(self, metrics: Optional[Dict[str, float]] = None):
        """
        打印指标

        Args:
            metrics: 指标字典（可选）
        """
        if metrics is None:
            metrics = self.compute_metrics()

        if not metrics:
            print("没有可用的指标数据")
            return

        print("\n=== 异常检测指标 ===")
        print(f"AUC-ROC: {metrics.get('auc', 0):.4f}")
        print(f"AUC-PR:  {metrics.get('ap', 0):.4f}")
        print(f"F1分数:  {metrics.get('f1', 0):.4f}")
        print(f"精确率:  {metrics.get('precision', 0):.4f}")
        print(f"召回率:  {metrics.get('recall', 0):.4f}")
        print(f"准确率:  {metrics.get('accuracy', 0):.4f}")
        print(f"特异性:  {metrics.get('specificity', 0):.4f}")
        print(f"敏感性:  {metrics.get('sensitivity', 0):.4f}")
        print(f"平衡准确率: {metrics.get('balanced_accuracy', 0):.4f}")
        print(f"最优阈值: {metrics.get('optimal_threshold', 0.5):.4f}")


class PerPixelMetrics:
    """
    像素级异常检测指标（用于分割任务）
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置累积指标"""
        self.all_predictions = []
        self.all_labels = []

    def update(self, predictions: Union[np.ndarray, torch.Tensor],
               labels: Union[np.ndarray, torch.Tensor]):
        """
        更新累积指标

        Args:
            predictions: 预测mask [B, H, W] 或 [B, 1, H, W]
            labels: 真实mask [B, H, W] 或 [B, 1, H, W]
        """
        # 转换为numpy并展平
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        # 确保形状一致
        if predictions.ndim == 4:
            predictions = predictions.squeeze(1)
        if labels.ndim == 4:
            labels = labels.squeeze(1)

        self.all_predictions.extend(predictions.flatten())
        self.all_labels.extend(labels.flatten())

    def compute_metrics(self) -> Dict[str, float]:
        """计算像素级指标"""
        if not self.all_predictions or not self.all_labels:
            return {}

        pred_array = np.array(self.all_predictions)
        label_array = np.array(self.all_labels)

        # 使用AnomalyMetrics计算基础指标
        metrics_calculator = AnomalyMetrics()
        metrics = metrics_calculator.compute_metrics(pred_array, label_array)

        # 添加像素级特定指标
        try:
            # IoU (Intersection over Union)
            intersection = np.sum((pred_array > 0.5) & (label_array > 0.5))
            union = np.sum((pred_array > 0.5) | (label_array > 0.5))
            metrics['iou'] = intersection / union if union > 0 else 0

            # Dice coefficient
            dice = 2 * intersection / (np.sum(pred_array > 0.5) + np.sum(label_array > 0.5))
            metrics['dice'] = dice if not np.isnan(dice) else 0

        except:
            metrics['iou'] = 0
            metrics['dice'] = 0

        return metrics


# 测试函数
def test_metrics():
    """测试指标计算"""
    print("=== 测试异常检测指标 ===")

    try:
        # 创建测试数据
        np.random.seed(42)
        n_samples = 1000

        # 模拟预测分数和标签
        normal_scores = np.random.beta(2, 5, n_samples // 2)  # 正常样本倾向于低分
        anomaly_scores = np.random.beta(5, 2, n_samples // 2)  # 异常样本倾向于高分

        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

        # 打乱数据
        indices = np.random.permutation(n_samples)
        scores = scores[indices]
        labels = labels[indices]

        print(f"测试数据: {n_samples} 样本")
        print(f"正常样本: {np.sum(labels == 0)}")
        print(f"异常样本: {np.sum(labels == 1)}")

        # 测试异常检测指标
        print("\n1. 测试异常检测指标")
        metrics_calculator = AnomalyMetrics()

        # 测试批量更新
        batch_size = 100
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            metrics_calculator.update(
                scores[i:end_idx],
                labels[i:end_idx]
            )

        # 计算最终指标
        metrics = metrics_calculator.compute_metrics()
        metrics_calculator.print_metrics(metrics)

        # 测试单次计算
        print("\n2. 测试单次指标计算")
        metrics_single = metrics_calculator.compute_metrics(scores, labels)
        print(f"AUC (单次): {metrics_single['auc']:.4f}")
        print(f"F1 (单次): {metrics_single['f1']:.4f}")

        # 测试ROC和PR曲线数据
        print("\n3. 测试曲线数据")
        fpr, tpr, roc_thresholds = metrics_calculator.get_roc_curve_data(scores, labels)
        precision, recall, pr_thresholds = metrics_calculator.get_pr_curve_data(scores, labels)

        print(f"ROC曲线点数: {len(fpr)}")
        print(f"PR曲线点数: {len(precision)}")

        # 测试像素级指标
        print("\n4. 测试像素级指标")
        pixel_metrics = PerPixelMetrics()

        # 创建模拟的分割mask
        pred_masks = np.random.rand(10, 64, 64) > 0.7
        true_masks = np.random.rand(10, 64, 64) > 0.8

        pixel_metrics.update(pred_masks, true_masks)
        pixel_results = pixel_metrics.compute_metrics()

        print(f"像素级IoU: {pixel_results.get('iou', 0):.4f}")
        print(f"像素级Dice: {pixel_results.get('dice', 0):.4f}")
        print(f"像素级AUC: {pixel_results.get('auc', 0):.4f}")

        print("\n🎉 指标测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_metrics()