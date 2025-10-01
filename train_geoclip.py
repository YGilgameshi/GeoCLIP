"""
GeoCLIP Training Script
基于AdaCLIP的训练脚本，添加了3D特征处理
"""

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

# 添加AdaCLIP路径到Python路径 (用于导入AdaCLIP的数据集)
current_dir = Path(__file__).parent.absolute()
if current_dir not in sys.path:
    sys.path.insert(0, str(current_dir))

# GeoCLIP组件导入
from geoclip.models.geoclip_main import create_geoclip_model
from geoclip.training.trainer import create_trainer
from geoclip.training.losses import create_loss_function
from geoclip.datasets.adaclip_adapter import create_geoclip_dataset
from geoclip.utils.metrics import AnomalyMetrics
from geoclip.utils.visualization import Visualizer

# AdaCLIP工具导入 (从复制的文件导入)
try:
    from tools.tools import setup_seed, Logger
except ImportError:
    print("警告: 无法导入AdaCLIP工具，将使用GeoCLIP内置功能")


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)


    class Logger:
        def __init__(self, log_path):
            self.log_path = log_path
            if log_path:
                # 清空日志文件
                with open(log_path, 'w') as f:
                    f.write("")

        def info(self, msg):
            print(msg)
            if self.log_path:
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(msg + '\n')

# 确保能找到AdaCLIP复制的数据集
try:
    from dataset import get_data

    ADACLIP_DATA_AVAILABLE = True
except ImportError:
    print("警告: AdaCLIP数据集函数不可用，将使用GeoCLIP数据集接口")
    ADACLIP_DATA_AVAILABLE = False


def str2bool(v):
    """字符串转布尔值"""
    return v.lower() in ("yes", "true", "t", "1")


def create_data_loaders(args, model_transforms=None):
    """
    创建数据加载器，兼容AdaCLIP的数据集结构
    """
    print("🔄 创建数据加载器...")

    # 设置默认transforms
    if model_transforms is None:
        import torchvision.transforms as transforms
        model_transforms = {
            'preprocess': transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            'transform': transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor()
            ])
        }

    # 创建训练数据集
    train_datasets = []
    for dataset_name in args.training_data:
        try:
            print(f"📊 加载训练数据集: {dataset_name}")

            # 对于MVTec等大数据集，可以选择性地预处理深度图
            preprocess_depth = args.preprocess_depth and dataset_name in ['mvtec', 'visa']

            train_dataset = create_geoclip_dataset(
                dataset_name=dataset_name,
                clsnames=None,  # 使用默认类别
                transform=model_transforms['preprocess'],
                target_transform=model_transforms['transform'],
                depth_transform=model_transforms['transform'],
                training=True,
                cache_depth=True,
                preprocess_all=preprocess_depth,
                depth_cache_dir=f'./depth_cache/{dataset_name}'
            )

            train_datasets.append(train_dataset)
            print(f"✅ {dataset_name} 训练集: {len(train_dataset)} 样本")

        except Exception as e:
            print(f"❌ 加载 {dataset_name} 训练集失败: {e}")
            continue

    # 合并训练数据集
    if len(train_datasets) > 1:
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_datasets)
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        raise ValueError("无法创建任何训练数据集")

    # 创建测试数据集 (用作验证)
    test_datasets = []
    for dataset_name in args.testing_data:
        try:
            print(f"📊 加载测试数据集: {dataset_name}")

            test_dataset = create_geoclip_dataset(
                dataset_name=dataset_name,
                clsnames=None,
                transform=model_transforms['preprocess'],
                target_transform=model_transforms['transform'],
                depth_transform=model_transforms['transform'],
                training=False,
                cache_depth=True,
                preprocess_all=False,  # 测试集不预处理
                depth_cache_dir=f'./depth_cache/{dataset_name}'
            )

            test_datasets.append(test_dataset)
            print(f"✅ {dataset_name} 测试集: {len(test_dataset)} 样本")

        except Exception as e:
            print(f"❌ 加载 {dataset_name} 测试集失败: {e}")
            continue

    # 合并测试数据集
    if len(test_datasets) > 1:
        from torch.utils.data import ConcatDataset
        test_dataset = ConcatDataset(test_datasets)
    elif len(test_datasets) == 1:
        test_dataset = test_datasets[0]
    else:
        raise ValueError("无法创建任何测试数据集")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"📊 数据加载器创建完成:")
    print(f"   训练样本: {len(train_dataset)}")
    print(f"   测试样本: {len(test_dataset)}")
    print(f"   批次大小: {args.batch_size}")

    return train_loader, test_loader


def setup_experiment_paths(args):
    """设置实验路径"""
    # 创建实验名称
    train_data_str = '_'.join(args.training_data)
    test_data_str = '_'.join(args.testing_data) if isinstance(args.testing_data, list) else args.testing_data

    experiment_name = f"geoclip_{train_data_str}_vs_{test_data_str}_{args.model.replace('/', '-')}"

    # 创建路径
    save_path = Path(args.save_path) / experiment_name
    save_path.mkdir(parents=True, exist_ok=True)

    paths = {
        'experiment_name': experiment_name,
        'save_path': save_path,
        'log_path': save_path / 'train.log',
        'config_path': save_path / 'config.json',
        'results_path': save_path / 'results',
        'checkpoints_path': save_path / 'checkpoints',
        'visualizations_path': save_path / 'visualizations'
    }

    # 创建子目录
    for key in ['results_path', 'checkpoints_path', 'visualizations_path']:
        paths[key].mkdir(exist_ok=True)

    return paths


def create_model_config(args):
    """创建GeoCLIP模型配置"""
    model_config = {
        # CLIP配置
        'clip_model': args.model,
        'clip_pretrained': 'openai',
        'freeze_clip': args.freeze_clip,

        # 深度估计配置
        'depth_estimator': 'DPT_Large',

        # 几何编码器配置
        'geometry_encoder': {
            'type': args.geometry_encoder,
            'in_channels': 4,  # RGB + Depth
            'base_channels': 64,
            'num_stages': 4,
            'output_channels': 512,
            'voxel_size': 64
        },

        # 特征融合配置
        'fusion': {
            'type': args.fusion_type,
            'clip_dim': 512,  # 会根据实际CLIP模型调整
            'geometry_dim': 512,
            'fusion_dim': args.fusion_dim,
            'output_dim': 512
        },

        # 异常检测配置
        'detection_type': 'regression',
        'num_classes': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    return model_config


def create_training_config(args):
    """创建训练配置"""
    training_config = {
        # 损失函数配置
        'loss': {
            'type': 'geoclip',
            'contrastive_weight': args.contrastive_weight,
            'geometry_weight': args.geometry_weight,
            'anomaly_weight': args.anomaly_weight,
            'use_contrastive': True,
            'use_geometry': True,
            'anomaly_config': {
                'loss_type': 'focal',
                'alpha': 0.25,
                'gamma': 2.0
            }
        },

        # 优化器配置
        'optimizer': {
            'type': 'adamw',
            'params': {
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay
            }
        },

        # 学习率调度器配置
        'scheduler': {
            'type': 'cosine',
            'params': {
                'T_max': args.epoch
            }
        },

        # 训练器配置
        'trainer': {
            'experiment_name': None,  # 会在后面设置
            'experiment_dir': str(args.save_path),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'use_wandb': args.use_wandb,
            'wandb_config': {
                'project': 'GeoCLIP',
                'entity': args.wandb_entity
            } if args.use_wandb else {},
            'log_interval': args.print_freq,
            'save_interval': args.save_freq,
            'early_stopping_patience': args.early_stopping_patience
        }
    }

    return training_config


def train_geoclip(args):
    """主训练函数"""
    print("🚀 开始GeoCLIP训练")
    print("=" * 50)

    # 设置随机种子
    setup_seed(args.seed)

    # 设置实验路径
    paths = setup_experiment_paths(args)

    # 设置日志
    logger = Logger(str(paths['log_path']))
    logger.info("开始GeoCLIP训练")

    # 打印配置信息
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    # 创建模型和训练配置
    model_config = create_model_config(args)
    training_config = create_training_config(args)
    training_config['trainer']['experiment_name'] = paths['experiment_name']

    # 保存配置
    full_config = {
        'model': model_config,
        'training': training_config,
        'args': vars(args)
    }

    with open(paths['config_path'], 'w') as f:
        json.dump(full_config, f, indent=2)

    logger.info(f"配置已保存: {paths['config_path']}")

    try:
        # 创建数据加载器
        train_loader, test_loader = create_data_loaders(args)

        # 创建训练器
        logger.info("创建GeoCLIP训练器...")
        trainer = create_trainer(
            config={
                'model': model_config,
                'loss': training_config['loss'],
                'optimizer': training_config['optimizer'],
                'scheduler': training_config['scheduler'],
                'trainer': training_config['trainer']
            },
            train_loader=train_loader,
            val_loader=test_loader,  # 使用测试集作为验证集
            test_loader=test_loader
        )

        logger.info("✅ 训练器创建成功")
        logger.info(f"模型信息: {trainer.model.get_model_info()}")

        # 开始训练
        logger.info(f"开始训练，共 {args.epoch} 个epoch...")
        test_results = trainer.train(
            num_epochs=args.epoch,
            resume_from=args.ckt_path if args.ckt_path else None
        )

        # 保存最终结果
        results_file = paths['results_path'] / 'final_results.json'
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)

        logger.info(f"✅ 训练完成！")
        logger.info(f"最终结果已保存: {results_file}")

        return test_results

    except Exception as e:
        logger.info(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser("GeoCLIP Training", add_help=True)

    # 数据集配置
    parser.add_argument("--training_data", type=str, default=["mvtec", "colondb"], nargs='+',
                        help="训练数据集列表")
    parser.add_argument("--testing_data", type=str, default=["visa"], nargs='+',
                        help="测试数据集列表")

    # 路径配置
    parser.add_argument("--save_path", type=str, default='./workspaces_geoclip',
                        help="结果保存目录")

    # 模型配置
    parser.add_argument("--model", type=str, default="ViT-B-16",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="CLIP模型选择")

    parser.add_argument("--geometry_encoder", type=str, default="voxel",
                        choices=["voxel", "sparse", "hierarchical"],
                        help="几何编码器类型")

    parser.add_argument("--fusion_type", type=str, default="cross_attention",
                        choices=["cross_attention", "adaptive_fusion", "simple_concat", "gated_fusion"],
                        help="特征融合类型")

    parser.add_argument("--fusion_dim", type=int, default=1024,
                        help="融合层维度")

    parser.add_argument("--freeze_clip", type=str2bool, default=False,
                        help="是否冻结CLIP参数")

    # 训练配置
    parser.add_argument("--epoch", type=int, default=100, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--image_size", type=int, default=224, help="图像尺寸")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")

    # 损失函数权重
    parser.add_argument("--contrastive_weight", type=float, default=1.0, help="对比学习损失权重")
    parser.add_argument("--geometry_weight", type=float, default=0.5, help="几何一致性损失权重")
    parser.add_argument("--anomaly_weight", type=float, default=2.0, help="异常检测损失权重")

    # 深度处理配置
    parser.add_argument("--preprocess_depth", type=str2bool, default=True,
                        help="是否预处理深度图缓存")

    # 训练控制
    parser.add_argument("--print_freq", type=int, default=10, help="日志打印频率")
    parser.add_argument("--save_freq", type=int, default=5, help="模型保存频率")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="早停耐心值")

    # 实验跟踪
    parser.add_argument("--use_wandb", type=str2bool, default=False, help="是否使用wandb")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb实体名")

    # 其他配置
    parser.add_argument("--seed", type=int, default=111, help="随机种子")
    parser.add_argument("--ckt_path", type=str, default=None, help="恢复训练的检查点路径")

    args = parser.parse_args()

    # 开始训练
    results = train_geoclip(args)

    if results is not None:
        print("\n🎉 GeoCLIP训练成功完成！")
        print(f"最佳测试结果:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
    else:
        print("\n❌ GeoCLIP训练失败")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)