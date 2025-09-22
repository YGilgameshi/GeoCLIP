"""
GeoCLIP 基础数据集类
扩展原有BaseDataset以支持3D数据
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
import sys
import os

# 添加AdaCLIP数据集路径到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
adaclip_dataset_path = os.path.join(parent_dir, 'dataset')
if adaclip_dataset_path not in sys.path:
    sys.path.insert(0, adaclip_dataset_path)

try:
    from base_dataset import BaseDataset as AdaCLIPBaseDataset
except ImportError:
    # Fallback: 如果无法导入，创建一个简单的基类
    print("警告: 无法导入AdaCLIP BaseDataset，使用fallback实现")


    class AdaCLIPBaseDataset(Dataset):
        def __init__(self, clsnames, transform, target_transform, root,
                     aug_rate=0., training=True):
            self.clsnames = clsnames
            self.transform = transform
            self.target_transform = target_transform
            self.root = root
            self.aug_rate = aug_rate
            self.training = training
            self.data_all = []
            self.length = 0

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            # 这是一个fallback实现，实际使用中应该从AdaCLIP导入
            raise NotImplementedError("请确保正确导入AdaCLIP BaseDataset")


class GeoCLIPBaseDataset(AdaCLIPBaseDataset):
    """
    GeoCLIP基础数据集类
    扩展AdaCLIP BaseDataset以支持3D数据
    """

    def __init__(self,
                 clsnames,
                 transform,
                 target_transform,
                 root,
                 aug_rate=0.,
                 training=True,
                 depth_transform=None,
                 **kwargs):

        # 调用父类初始化
        super().__init__(
            clsnames=clsnames,
            transform=transform,
            target_transform=target_transform,
            root=root,
            aug_rate=aug_rate,
            training=training
        )

        self.depth_transform = depth_transform
        self.enable_3d = True  # 标识这是3D数据集

    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """获取样本基本信息（不加载实际数据）"""
        if index >= len(self.data_all):
            raise IndexError(f"索引 {index} 超出数据集大小 {len(self.data_all)}")

        return self.data_all[index].copy()

    def has_depth(self, index: int) -> bool:
        """检查样本是否有深度信息"""
        # 子类应该重写这个方法
        return True

    def get_depth_info(self, index: int) -> Dict[str, Any]:
        """获取深度信息"""
        # 子类应该重写这个方法
        return {
            'has_depth': self.has_depth(index),
            'depth_range': (0.0, 1.0),
            'depth_quality': 'estimated'
        }

    def validate_sample(self, sample: Dict[str, Any]) -> bool:
        """验证样本数据的完整性"""
        required_keys = ['img', 'cls_name', 'anomaly']

        # 检查必需的键
        for key in required_keys:
            if key not in sample:
                return False

        # 检查3D数据特有的键
        if self.enable_3d:
            if 'depth' not in sample:
                return False

        # 检查数据类型和形状
        if hasattr(sample['img'], 'shape') and len(sample['img'].shape) != 3:
            return False

        if 'depth' in sample:
            if hasattr(sample['depth'], 'shape'):
                depth_shape = sample['depth'].shape
                if len(depth_shape) < 2 or len(depth_shape) > 3:
                    return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_samples': len(self.data_all),
            'num_classes': len(self.clsnames),
            'class_names': self.clsnames.copy(),
            'is_3d': self.enable_3d,
            'training': self.training
        }

        # 统计每个类别的样本数
        class_counts = {}
        anomaly_counts = {'normal': 0, 'anomaly': 0}

        for data in self.data_all:
            cls_name = data['cls_name']
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            if data['anomaly'] == 0:
                anomaly_counts['normal'] += 1
            else:
                anomaly_counts['anomaly'] += 1

        stats['class_distribution'] = class_counts
        stats['anomaly_distribution'] = anomaly_counts

        return stats

    def print_info(self):
        """打印数据集信息"""
        stats = self.get_statistics()

        print(f"\n=== GeoCLIP数据集信息 ===")
        print(f"数据集类型: {'3D (RGB+深度)' if stats['is_3d'] else '2D (仅RGB)'}")
        print(f"训练模式: {stats['training']}")
        print(f"总样本数: {stats['total_samples']}")
        print(f"类别数量: {stats['num_classes']}")
        print(f"类别名称: {stats['class_names']}")

        print(f"\n类别分布:")
        for cls_name, count in stats['class_distribution'].items():
            print(f"  {cls_name}: {count} 样本")

        print(f"\n异常分布:")
        total = stats['anomaly_distribution']['normal'] + stats['anomaly_distribution']['anomaly']
        normal_pct = stats['anomaly_distribution']['normal'] / total * 100
        anomaly_pct = stats['anomaly_distribution']['anomaly'] / total * 100
        print(f"  正常样本: {stats['anomaly_distribution']['normal']} ({normal_pct:.1f}%)")
        print(f"  异常样本: {stats['anomaly_distribution']['anomaly']} ({anomaly_pct:.1f}%)")


# 测试代码
if __name__ == "__main__":
    print("测试GeoCLIP基础数据集类...")


    # 创建一个简单的测试数据集
    class TestDataset(GeoCLIPBaseDataset):
        def __init__(self):
            super().__init__(
                clsnames=['test_class'],
                transform=None,
                target_transform=None,
                root='./test',
                training=True
            )

            # 添加一些测试数据
            self.data_all = [
                {'cls_name': 'test_class', 'anomaly': 0, 'img_path': 'test1.jpg'},
                {'cls_name': 'test_class', 'anomaly': 1, 'img_path': 'test2.jpg'},
            ]
            self.length = len(self.data_all)

        def __getitem__(self, index):
            # 简单的测试实现
            sample = self.get_sample_info(index)
            sample['img'] = torch.randn(3, 224, 224)  # 假图像
            sample['depth'] = torch.randn(1, 224, 224)  # 假深度
            sample['has_depth'] = True
            return sample


    # 测试
    dataset = TestDataset()
    dataset.print_info()

    # 测试样本获取
    sample = dataset[0]
    print(f"\n样本测试:")
    print(f"样本键: {list(sample.keys())}")
    print(f"样本验证: {dataset.validate_sample(sample)}")