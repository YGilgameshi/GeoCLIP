"""
GeoCLIP - 训练器模块
端到端的训练和验证流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import wandb

# 导入GeoCLIP组件
from geoclip.models.geoclip_main import GeoCLIP, create_geoclip_model
from geoclip.training.losses import GeoCLIPLoss, create_loss_function
from geoclip.utils.metrics import AnomalyMetrics
from geoclip.utils.visualization import Visualizer


class EarlyStopping:
    """早停机制"""

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.001,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class GeoCLIPTrainer:
    """
    GeoCLIP训练器
    管理整个训练和验证流程
    """

    def __init__(self,
                 model: GeoCLIP,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 loss_function: Optional[nn.Module] = None,
                 optimizer: Optional[optim.Optimizer] = None,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda',
                 experiment_dir: str = './experiments',
                 experiment_name: str = 'geoclip_exp',
                 use_wandb: bool = False,
                 wandb_config: Dict = None,
                 log_interval: int = 10,
                 save_interval: int = 5,
                 early_stopping_patience: int = 15):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.log_interval = log_interval
        self.save_interval = save_interval

        # 设置实验目录
        self.experiment_dir = Path(experiment_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 损失函数
        if loss_function is None:
            self.loss_function = GeoCLIPLoss()
        else:
            self.loss_function = loss_function

        # 优化器
        if optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-5
            )
        else:
            self.optimizer = optimizer

        # 学习率调度器
        self.scheduler = scheduler

        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max'  # 监控验证AUC，越大越好
        )

        # 指标计算
        self.metrics = AnomalyMetrics()

        # 可视化工具
        self.visualizer = Visualizer(save_dir=self.experiment_dir / 'visualizations')

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.experiment_dir / 'tensorboard')

        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get('project', 'GeoCLIP'),
                name=experiment_name,
                config=wandb_config,
                dir=str(self.experiment_dir)
            )
            wandb.watch(self.model, log_freq=100)

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_auc = 0.0
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)

        print(f"GeoCLIP训练器初始化完成")
        print(f"实验目录: {self.experiment_dir}")
        print(f"设备: {device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)

        progress_bar = tqdm(self.train_loader, desc=f'训练 Epoch {self.current_epoch}')

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = self._move_to_device(batch)

            # 前向传播
            self.optimizer.zero_grad()

            try:
                model_outputs = self.model(batch)

                # 准备目标
                targets = {
                    'anomaly_labels': batch['anomaly'],
                    'class_labels': batch.get('cls_name', None)
                }

                # 计算损失
                loss_dict = self.loss_function(model_outputs, targets)
                total_loss = loss_dict['total_loss']

                # 反向传播
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # 记录损失
                for key, value in loss_dict.items():
                    epoch_losses[key].append(value.item())

                # 计算指标
                if 'anomaly_predictions' in model_outputs:
                    predictions = model_outputs['anomaly_predictions']
                    if predictions.dim() > 1:
                        predictions = predictions[:, 1]  # 取正类概率

                    labels = targets['anomaly_labels']

                    # 转换为numpy
                    pred_np = predictions.detach().cpu().numpy()
                    label_np = labels.detach().cpu().numpy()

                    # 计算批次指标
                    batch_metrics = self.metrics.compute_metrics(pred_np, label_np)
                    for key, value in batch_metrics.items():
                        if not np.isnan(value):
                            epoch_metrics[key].append(value)

                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'auc': f"{np.mean(epoch_metrics['auc']) if epoch_metrics['auc'] else 0:.3f}"
                })

                # 记录到TensorBoard
                if self.global_step % self.log_interval == 0:
                    self._log_to_tensorboard('train', loss_dict, self.global_step)

                    if self.use_wandb:
                        wandb_log = {f'train/{k}': v.item() if torch.is_tensor(v) else v
                                     for k, v in loss_dict.items()}
                        wandb.log(wandb_log, step=self.global_step)

                self.global_step += 1

            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue

        # 计算平均指标
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}

        return {**avg_losses, **avg_metrics}

    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        epoch_losses = defaultdict(list)
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(self.val_loader, desc=f'验证 Epoch {self.current_epoch}')

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # 移动数据到设备
                batch = self._move_to_device(batch)

                try:
                    # 前向传播
                    model_outputs = self.model(batch)

                    # 准备目标
                    targets = {
                        'anomaly_labels': batch['anomaly'],
                        'class_labels': batch.get('cls_name', None)
                    }

                    # 计算损失
                    loss_dict = self.loss_function(model_outputs, targets)

                    # 记录损失
                    for key, value in loss_dict.items():
                        epoch_losses[key].append(value.item())

                    # 收集预测和标签
                    if 'anomaly_predictions' in model_outputs:
                        predictions = model_outputs['anomaly_predictions']
                        if predictions.dim() > 1:
                            predictions = predictions[:, 1]  # 取正类概率

                        labels = targets['anomaly_labels']

                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                    # 更新进度条
                    progress_bar.set_postfix({
                        'val_loss': f"{loss_dict['total_loss'].item():.4f}"
                    })

                except Exception as e:
                    print(f"验证批次 {batch_idx} 出错: {e}")
                    continue

        # 计算平均损失
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}

        # 计算整体指标
        if all_predictions and all_labels:
            val_metrics = self.metrics.compute_metrics(
                np.array(all_predictions),
                np.array(all_labels)
            )
        else:
            val_metrics = {}

        return {**avg_losses, **val_metrics}

    def test(self) -> Dict[str, float]:
        """测试模型"""
        if self.test_loader is None:
            print("警告: 没有提供测试数据集")
            return {}

        print("开始测试...")
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_features = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='测试'):
                batch = self._move_to_device(batch)

                try:
                    model_outputs = self.model(batch)

                    if 'anomaly_predictions' in model_outputs:
                        predictions = model_outputs['anomaly_predictions']
                        if predictions.dim() > 1:
                            predictions = predictions[:, 1]

                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(batch['anomaly'].cpu().numpy())

                        # 收集特征用于可视化
                        if 'fused_features' in model_outputs:
                            all_features.extend(model_outputs['fused_features'].cpu().numpy())

                except Exception as e:
                    print(f"测试批次出错: {e}")
                    continue

        # 计算测试指标
        if all_predictions and all_labels:
            test_metrics = self.metrics.compute_metrics(
                np.array(all_predictions),
                np.array(all_labels)
            )

            # 保存测试结果
            test_results = {
                'predictions': all_predictions,
                'labels': all_labels,
                'metrics': test_metrics
            }

            results_path = self.experiment_dir / 'test_results.json'
            with open(results_path, 'w') as f:
                # 转换numpy数组为列表以便JSON序列化
                serializable_results = {
                    'predictions': [float(x) for x in all_predictions],
                    'labels': [int(x) for x in all_labels],
                    'metrics': {k: float(v) for k, v in test_metrics.items()}
                }
                json.dump(serializable_results, f, indent=2)

            print(f"测试完成，结果保存到: {results_path}")
            print("测试指标:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")

            # 可视化测试结果
            if all_features:
                self.visualizer.plot_feature_distribution(
                    np.array(all_features),
                    np.array(all_labels),
                    save_name='test_feature_distribution.png'
                )

                self.visualizer.plot_roc_curve(
                    np.array(all_labels),
                    np.array(all_predictions),
                    save_name='test_roc_curve.png'
                )
        else:
            test_metrics = {}
            print("测试失败: 没有收集到有效的预测结果")

        return test_metrics

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """完整的训练流程"""
        # 恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"从检查点恢复训练: {resume_from}")

        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset)}")

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # 训练
            train_results = self.train_epoch()
            self.train_history['epoch'].append(epoch)
            for key, value in train_results.items():
                self.train_history[key].append(value)

            # 验证
            val_results = self.validate_epoch()
            self.val_history['epoch'].append(epoch)
            for key, value in val_results.items():
                self.val_history[key].append(value)

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results.get('auc', 0))
                else:
                    self.scheduler.step()

            # 记录到TensorBoard和wandb
            current_lr = self.optimizer.param_groups[0]['lr']
            self._log_to_tensorboard('val', val_results, epoch)
            self.writer.add_scalar('learning_rate', current_lr, epoch)

            if self.use_wandb:
                wandb_log = {
                    'epoch': epoch,
                    'learning_rate': current_lr,
                    **{f'train/{k}': v for k, v in train_results.items()},
                    **{f'val/{k}': v for k, v in val_results.items()}
                }
                wandb.log(wandb_log, step=epoch)

            # 打印进度
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"训练 - Loss: {train_results.get('total_loss', 0):.4f}, "
                  f"AUC: {train_results.get('auc', 0):.4f}")
            print(f"验证 - Loss: {val_results.get('total_loss', 0):.4f}, "
                  f"AUC: {val_results.get('auc', 0):.4f}")

            # 保存最佳模型
            val_auc = val_results.get('auc', 0)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.save_checkpoint('best_model.pth', is_best=True)
                print(f"✅ 保存最佳模型 (AUC: {val_auc:.4f})")

            # 定期保存检查点
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # 早停检查
            if self.early_stopping(val_auc):
                print(f"早停触发，在第 {epoch + 1} 个epoch停止训练")
                break

        # 训练完成
        total_time = time.time() - start_time
        print(f"\n训练完成! 总用时: {total_time / 3600:.2f} 小时")
        print(f"最佳验证AUC: {self.best_val_auc:.4f}")

        # 保存训练历史
        self._save_training_history()

        # 生成训练报告
        self._generate_training_report()

        # 在最佳模型上进行测试
        print("\n在最佳模型上进行测试...")
        self.load_checkpoint(self.experiment_dir / 'best_model.pth')
        test_results = self.test()

        return test_results

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = self.experiment_dir / filename
        torch.save(checkpoint, save_path)

        if is_best:
            # 同时保存为latest.pth
            torch.save(checkpoint, self.experiment_dir / 'latest.pth')

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """加载检查点"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_auc = checkpoint['best_val_auc']
        self.train_history = defaultdict(list, checkpoint['train_history'])
        self.val_history = defaultdict(list, checkpoint['val_history'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"成功加载检查点: {checkpoint_path}")

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将批次数据移动到指定设备"""
        moved_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch

    def _log_to_tensorboard(self, phase: str, metrics: Dict[str, float], step: int):
        """记录到TensorBoard"""
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.item()
            self.writer.add_scalar(f'{phase}/{key}', value, step)

    def _save_training_history(self):
        """保存训练历史"""
        history = {
            'train': dict(self.train_history),
            'val': dict(self.val_history)
        }

        history_path = self.experiment_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # 绘制训练曲线
        self.visualizer.plot_training_curves(
            dict(self.train_history),
            dict(self.val_history),
            save_name='training_curves.png'
        )

    def _generate_training_report(self):
        """生成训练报告"""
        report = {
            'experiment_info': {
                'name': self.experiment_dir.name,
                'total_epochs': self.current_epoch + 1,
                'best_val_auc': self.best_val_auc,
                'final_lr': self.optimizer.param_groups[0]['lr']
            },
            'model_info': self.model.get_model_info(),
            'training_config': {
                'optimizer': type(self.optimizer).__name__,
                'scheduler': type(self.scheduler).__name__ if self.scheduler else None,
                'batch_size': self.train_loader.batch_size,
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset)
            }
        }

        report_path = self.experiment_dir / 'training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"训练报告已保存: {report_path}")


def create_trainer(config: Dict[str, Any],
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   test_loader: Optional[DataLoader] = None) -> GeoCLIPTrainer:
    """
    根据配置创建训练器

    Args:
        config: 训练配置
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器 (可选)

    Returns:
        trainer: GeoCLIP训练器实例
    """
    # 创建模型
    model_config = config.get('model', {})
    model = create_geoclip_model(model_config)

    # 创建损失函数
    loss_config = config.get('loss', {})
    loss_function = create_loss_function(loss_config)

    # 创建优化器
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adamw')
    optimizer_params = optimizer_config.get('params', {})

    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **optimizer_params)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

    # 创建学习率调度器
    scheduler = None
    if 'scheduler' in config:
        scheduler_config = config['scheduler']
        scheduler_type = scheduler_config.get('type', 'cosine')
        scheduler_params = scheduler_config.get('params', {})

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)

    # 创建训练器
    trainer_config = config.get('trainer', {})
    trainer = GeoCLIPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        **trainer_config
    )

    return trainer


# 使用示例和测试
def test_trainer():
    """测试训练器"""
    print("=== 测试GeoCLIP训练器 ===")

    try:
        from torch.utils.data import TensorDataset, DataLoader

        # 创建模拟数据集
        def create_mock_dataset(num_samples: int):
            images = torch.randn(num_samples, 3, 224, 224)
            depths = torch.rand(num_samples, 1, 224, 224) * 5
            anomalies = torch.randint(0, 2, (num_samples,))
            img_paths = [f'image_{i}.jpg' for i in range(num_samples)]

            dataset_dict = {
                'img': images,
                'depth': depths,
                'anomaly': anomalies,
                'img_path': img_paths,
                'has_depth': torch.ones(num_samples, dtype=torch.bool)
            }

            return dataset_dict

        # 创建数据集
        train_data = create_mock_dataset(100)
        val_data = create_mock_dataset(50)
        test_data = create_mock_dataset(30)

        # 转换为TensorDataset (简化版本)
        class MockDataset:
            def __init__(self, data_dict):
                self.data = data_dict
                self.length = len(data_dict['img'])

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                return {key: value[idx] if torch.is_tensor(value) else value[idx]
                        for key, value in self.data.items()}

        train_dataset = MockDataset(train_data)
        val_dataset = MockDataset(val_data)
        test_dataset = MockDataset(test_data)

        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 创建配置
        config = {
            'model': {
                'clip_model': 'ViT-B/16',
                'depth_estimator': 'DPT_Large',
                'fusion_type': 'cross_attention',
                'detection_type': 'regression',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'loss': {
                'type': 'geoclip',
                'contrastive_weight': 1.0,
                'geometry_weight': 0.5,
                'anomaly_weight': 2.0
            },
            'optimizer': {
                'type': 'adamw',
                'params': {
                    'lr': 1e-4,
                    'weight_decay': 1e-5
                }
            },
            'scheduler': {
                'type': 'cosine',
                'params': {
                    'T_max': 10
                }
            },
            'trainer': {
                'experiment_name': 'test_geoclip',
                'experiment_dir': './test_experiments',
                'use_wandb': False,
                'log_interval': 5,
                'save_interval': 2,
                'early_stopping_patience': 5
            }
        }

        # 创建训练器
        trainer = create_trainer(config, train_loader, val_loader, test_loader)
        print(f"✅ 训练器创建成功")

        # 测试训练循环 (少量epoch)
        print("\n开始测试训练...")
        test_results = trainer.train(num_epochs=3)

        print(f"✅ 训练测试完成")
        print(f"测试结果: {test_results}")

        print("\n🎉 训练器测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_trainer()