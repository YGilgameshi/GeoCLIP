"""
GeoCLIP - 可视化工具
结果可视化和分析工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Visualizer:
    """
    GeoCLIP可视化工具类
    """

    def __init__(self, save_dir: str = './visualizations', dpi: int = 300):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")

    def plot_training_curves(self,
                             train_history: Dict[str, List[float]],
                             val_history: Dict[str, List[float]],
                             save_name: str = 'training_curves.png'):
        """
        绘制训练曲线

        Args:
            train_history: 训练历史
            val_history: 验证历史
            save_name: 保存文件名
        """
        # 获取所有指标名称
        metrics = set(train_history.keys()) & set(val_history.keys())
        metrics = [m for m in metrics if m != 'epoch' and len(train_history[m]) > 0]

        if not metrics:
            print("没有可绘制的指标")
            return

        # 计算子图布局
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        epochs = range(1, len(train_history[metrics[0]]) + 1)

        for i, metric in enumerate(metrics):
            ax = axes[i] if i < len(axes) else plt.subplot(n_rows, n_cols, i + 1)

            # 绘制训练和验证曲线
            if metric in train_history and len(train_history[metric]) > 0:
                ax.plot(epochs[:len(train_history[metric])], train_history[metric],
                        label=f'训练 {metric}', marker='o', markersize=3)

            if metric in val_history and len(val_history[metric]) > 0:
                ax.plot(epochs[:len(val_history[metric])], val_history[metric],
                        label=f'验证 {metric}', marker='s', markersize=3)

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.upper()} 曲线')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存: {self.save_dir / save_name}")

    def plot_roc_curve(self,
                       labels: np.ndarray,
                       scores: np.ndarray,
                       save_name: str = 'roc_curve.png'):
        """
        绘制ROC曲线

        Args:
            labels: 真实标签
            scores: 预测分数
            save_name: 保存文件名
        """
        try:
            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC曲线 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                     label='随机分类器')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率 (FPR)')
            plt.ylabel('真阳性率 (TPR)')
            plt.title('ROC曲线')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)

            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"ROC曲线已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制ROC曲线失败: {e}")

    def plot_pr_curve(self,
                      labels: np.ndarray,
                      scores: np.ndarray,
                      save_name: str = 'pr_curve.png'):
        """
        绘制Precision-Recall曲线

        Args:
            labels: 真实标签
            scores: 预测分数
            save_name: 保存文件名
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score

            precision, recall, _ = precision_recall_curve(labels, scores)
            ap_score = average_precision_score(labels, scores)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                     label=f'PR曲线 (AP = {ap_score:.3f})')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('召回率 (Recall)')
            plt.ylabel('精确率 (Precision)')
            plt.title('Precision-Recall曲线')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)

            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"PR曲线已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制PR曲线失败: {e}")

    def plot_confusion_matrix(self,
                              labels: np.ndarray,
                              predictions: np.ndarray,
                              class_names: Optional[List[str]] = None,
                              save_name: str = 'confusion_matrix.png'):
        """
        绘制混淆矩阵

        Args:
            labels: 真实标签
            predictions: 预测标签
            class_names: 类别名称
            save_name: 保存文件名
        """
        try:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(labels, predictions)

            if class_names is None:
                class_names = [f'类别{i}' for i in range(cm.shape[0])]

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title('混淆矩阵')

            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"混淆矩阵已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制混淆矩阵失败: {e}")

    def plot_feature_distribution(self,
                                  features: np.ndarray,
                                  labels: np.ndarray,
                                  method: str = 'tsne',
                                  save_name: str = 'feature_distribution.png'):
        """
        绘制特征分布图

        Args:
            features: 特征向量 [N, D]
            labels: 标签 [N]
            method: 降维方法 ('tsne' 或 'pca')
            save_name: 保存文件名
        """
        try:
            print(f"使用{method.upper()}进行特征可视化...")

            # 降维
            if method.lower() == 'tsne':
                if features.shape[0] > 1000:
                    # 对于大数据集，先用PCA降维
                    pca = PCA(n_components=50)
                    features_reduced = pca.fit_transform(features)
                else:
                    features_reduced = features

                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, features_reduced.shape[0] - 1))
                features_2d = tsne.fit_transform(features_reduced)

            elif method.lower() == 'pca':
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(features)
            else:
                raise ValueError(f"不支持的降维方法: {method}")

            # 绘制散点图
            plt.figure(figsize=(10, 8))

            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                mask = labels == label
                label_name = '异常' if label == 1 else '正常'
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c=[color], label=label_name, alpha=0.7, s=20)

            plt.xlabel(f'{method.upper()}-1')
            plt.ylabel(f'{method.upper()}-2')
            plt.title(f'特征分布图 ({method.upper()})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"特征分布图已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制特征分布图失败: {e}")

    def plot_anomaly_heatmap(self,
                             image: np.ndarray,
                             anomaly_map: np.ndarray,
                             save_name: str = 'anomaly_heatmap.png',
                             alpha: float = 0.6):
        """
        绘制异常热力图

        Args:
            image: 原始图像 [H, W, 3]
            anomaly_map: 异常分数图 [H, W]
            save_name: 保存文件名
            alpha: 热力图透明度
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原始图像
            axes[0].imshow(image)
            axes[0].set_title('原始图像')
            axes[0].axis('off')

            # 异常热力图
            im = axes[1].imshow(anomaly_map, cmap='hot', interpolation='bilinear')
            axes[1].set_title('异常热力图')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # 叠加图像
            axes[2].imshow(image)
            axes[2].imshow(anomaly_map, cmap='hot', alpha=alpha, interpolation='bilinear')
            axes[2].set_title('叠加显示')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"异常热力图已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制异常热力图失败: {e}")

    def plot_depth_visualization(self,
                                 image: np.ndarray,
                                 depth: np.ndarray,
                                 save_name: str = 'depth_visualization.png'):
        """
        绘制深度可视化

        Args:
            image: RGB图像 [H, W, 3]
            depth: 深度图 [H, W]
            save_name: 保存文件名
        """
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # RGB图像
            axes[0].imshow(image)
            axes[0].set_title('RGB图像')
            axes[0].axis('off')

            # 深度图
            im1 = axes[1].imshow(depth, cmap='plasma')
            axes[1].set_title('深度图')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

            # 深度直方图
            axes[2].hist(depth.flatten(), bins=50, alpha=0.7, color='blue')
            axes[2].set_xlabel('深度值')
            axes[2].set_ylabel('像素数量')
            axes[2].set_title('深度分布直方图')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"深度可视化已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制深度可视化失败: {e}")

    def plot_3d_voxels(self,
                       voxels: np.ndarray,
                       save_name: str = 'voxel_visualization.png',
                       threshold: float = 0.5):
        """
        绘制3D体素可视化

        Args:
            voxels: 体素数据 [D, H, W]
            save_name: 保存文件名
            threshold: 显示阈值
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 8))

            # 3D体素图
            ax1 = fig.add_subplot(121, projection='3d')

            # 找到非零体素
            filled = voxels > threshold
            x, y, z = np.where(filled)

            if len(x) > 0:
                ax1.scatter(x, y, z, c=voxels[filled], cmap='viridis', alpha=0.6)

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('3D体素可视化')

            # 体素切片
            ax2 = fig.add_subplot(122)

            # 显示中间切片
            mid_slice = voxels.shape[0] // 2
            im = ax2.imshow(voxels[mid_slice], cmap='viridis')
            ax2.set_title(f'体素切片 (z={mid_slice})')
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"体素可视化已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制3D体素可视化失败: {e}")

    def plot_loss_components(self,
                             loss_history: Dict[str, List[float]],
                             save_name: str = 'loss_components.png'):
        """
        绘制损失函数组件

        Args:
            loss_history: 损失历史，包含各个损失组件
            save_name: 保存文件名
        """
        try:
            # 过滤损失相关的键
            loss_keys = [k for k in loss_history.keys() if 'loss' in k.lower()]

            if not loss_keys:
                print("没有找到损失相关的数据")
                return

            plt.figure(figsize=(12, 8))
            epochs = range(1, len(loss_history[loss_keys[0]]) + 1)

            for loss_key in loss_keys:
                if len(loss_history[loss_key]) > 0:
                    plt.plot(epochs[:len(loss_history[loss_key])],
                             loss_history[loss_key],
                             label=loss_key.replace('_', ' ').title(),
                             marker='o', markersize=3)

            plt.xlabel('Epoch')
            plt.ylabel('Loss Value')
            plt.title('损失函数组件')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # 使用对数刻度

            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"损失组件图已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制损失组件图失败: {e}")

    def plot_model_comparison(self,
                              results: Dict[str, Dict[str, float]],
                              save_name: str = 'model_comparison.png'):
        """
        绘制模型对比图

        Args:
            results: 模型结果字典 {model_name: {metric: value}}
            save_name: 保存文件名
        """
        try:
            if not results:
                print("没有可比较的模型结果")
                return

            # 获取所有指标
            all_metrics = set()
            for model_results in results.values():
                all_metrics.update(model_results.keys())

            metrics = sorted(list(all_metrics))
            models = list(results.keys())

            # 创建对比图
            x = np.arange(len(metrics))
            width = 0.8 / len(models)

            fig, ax = plt.subplots(figsize=(12, 8))

            for i, model in enumerate(models):
                values = [results[model].get(metric, 0) for metric in metrics]
                ax.bar(x + i * width, values, width, label=model, alpha=0.8)

            ax.set_xlabel('指标')
            ax.set_ylabel('数值')
            ax.set_title('模型性能对比')
            ax.set_xticks(x + width * (len(models) - 1) / 2)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"模型对比图已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"绘制模型对比图失败: {e}")

    def create_summary_report(self,
                              metrics: Dict[str, float],
                              model_info: Dict[str, Any],
                              save_name: str = 'summary_report.png'):
        """
        创建总结报告

        Args:
            metrics: 模型指标
            model_info: 模型信息
            save_name: 保存文件名
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # 1. 主要指标雷达图
            main_metrics = ['auc', 'ap', 'f1', 'precision', 'recall', 'accuracy']
            values = [metrics.get(m, 0) for m in main_metrics]

            angles = np.linspace(0, 2 * np.pi, len(main_metrics), endpoint=False)
            values += values[:1]  # 闭合图形
            angles = np.concatenate([angles, [angles[0]]])

            ax1.plot(angles, values, 'o-', linewidth=2, label='模型性能')
            ax1.fill(angles, values, alpha=0.25)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels([m.upper() for m in main_metrics])
            ax1.set_ylim(0, 1)
            ax1.set_title('主要指标雷达图')
            ax1.grid(True)

            # 2. 指标柱状图
            metric_names = list(metrics.keys())[:8]  # 显示前8个指标
            metric_values = [metrics[m] for m in metric_names]

            bars = ax2.bar(metric_names, metric_values, alpha=0.7)
            ax2.set_title('详细指标')
            ax2.set_ylabel('数值')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

            # 给柱状图添加数值标签
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')

            # 3. 模型信息表格
            ax3.axis('tight')
            ax3.axis('off')

            info_data = []
            for key, value in model_info.items():
                if isinstance(value, dict):
                    continue  # 跳过嵌套字典
                info_data.append([key.replace('_', ' ').title(), str(value)])

            if info_data:
                table = ax3.table(cellText=info_data,
                                  colLabels=['属性', '值'],
                                  cellLoc='left',
                                  loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)

            ax3.set_title('模型信息')

            # 4. 性能总结
            ax4.axis('off')

            # 创建性能总结文本
            summary_text = f"""
性能总结

🎯 主要指标:
• AUC-ROC: {metrics.get('auc', 0):.3f}
• AUC-PR: {metrics.get('ap', 0):.3f}
• F1分数: {metrics.get('f1', 0):.3f}

📊 分类性能:
• 精确率: {metrics.get('precision', 0):.3f}
• 召回率: {metrics.get('recall', 0):.3f}
• 准确率: {metrics.get('accuracy', 0):.3f}

⚡ 模型规模:
• 总参数: {model_info.get('total_parameters', 'N/A'):,}
• 可训练参数: {model_info.get('trainable_parameters', 'N/A'):,}
            """

            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            plt.suptitle('GeoCLIP模型性能报告', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            print(f"总结报告已保存: {self.save_dir / save_name}")

        except Exception as e:
            print(f"创建总结报告失败: {e}")


# 测试函数
def test_visualizer():
    """测试可视化工具"""
    print("=== 测试可视化工具 ===")

    try:
        # 创建可视化器
        viz = Visualizer(save_dir='./test_visualizations')

        # 1. 测试训练曲线
        print("\n1. 测试训练曲线")
        train_history = {
            'total_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
            'auc': [0.6, 0.7, 0.8, 0.85, 0.9],
            'f1': [0.5, 0.6, 0.7, 0.75, 0.8]
        }
        val_history = {
            'total_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
            'auc': [0.55, 0.65, 0.75, 0.8, 0.85],
            'f1': [0.45, 0.55, 0.65, 0.7, 0.75]
        }

        viz.plot_training_curves(train_history, val_history)

        # 2. 测试ROC和PR曲线
        print("\n2. 测试ROC和PR曲线")
        np.random.seed(42)
        n_samples = 1000

        # 模拟数据
        normal_scores = np.random.beta(2, 5, n_samples // 2)
        anomaly_scores = np.random.beta(5, 2, n_samples // 2)
        scores = np.concatenate([normal_scores, anomaly_scores])
        labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

        # 打乱数据
        indices = np.random.permutation(n_samples)
        scores = scores[indices]
        labels = labels[indices]

        viz.plot_roc_curve(labels, scores)
        viz.plot_pr_curve(labels, scores)

        # 3. 测试特征分布
        print("\n3. 测试特征分布")
        features = np.random.randn(500, 50)  # 500个样本，50维特征
        feature_labels = np.random.randint(0, 2, 500)

        viz.plot_feature_distribution(features, feature_labels, method='pca')

        # 4. 测试异常热力图
        print("\n4. 测试异常热力图")
        image = np.random.rand(128, 128, 3)
        anomaly_map = np.random.rand(128, 128)

        viz.plot_anomaly_heatmap(image, anomaly_map)

        # 5. 测试深度可视化
        print("\n5. 测试深度可视化")
        depth_map = np.random.rand(128, 128) * 10

        viz.plot_depth_visualization(image, depth_map)

        # 6. 测试模型对比
        print("\n6. 测试模型对比")
        model_results = {
            'GeoCLIP': {'auc': 0.95, 'f1': 0.85, 'precision': 0.88},
            'Baseline': {'auc': 0.78, 'f1': 0.70, 'precision': 0.75},
            'CLIP-Only': {'auc': 0.82, 'f1': 0.74, 'precision': 0.79}
        }

        viz.plot_model_comparison(model_results)

        # 7. 测试总结报告
        print("\n7. 测试总结报告")
        metrics = {
            'auc': 0.95, 'ap': 0.92, 'f1': 0.85,
            'precision': 0.88, 'recall': 0.82, 'accuracy': 0.90
        }
        model_info = {
            'total_parameters': 25000000,
            'trainable_parameters': 23000000,
            'fusion_type': 'cross_attention',
            'detection_type': 'regression'
        }

        viz.create_summary_report(metrics, model_info)

        print("\n🎉 可视化工具测试完成!")
        print(f"所有图片已保存到: {viz.save_dir}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_visualizer()