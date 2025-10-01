"""
GeoCLIP 数据集工厂
统一的数据集创建接口
"""

from typing import Dict, List, Optional, Any
import torchvision.transforms as transforms
from .adaclip_adapter import create_geoclip_dataset


class DatasetFactory:
    """GeoCLIP数据集工厂类"""

    # 支持的数据集映射
    SUPPORTED_DATASETS = {
        'mvtec': {
            'name': 'MVTec AD',
            'type': '2D_to_3D',
            'classes': ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                        'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        },
        'visa': {
            'name': 'VisA',
            'type': '2D_to_3D',
            'classes': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
                        'pcb4', 'pipe_fryum']
        },
        'colondb': {
            'name': 'CVC-ColonDB',
            'type': '2D_to_3D',
            'classes': ['ColonDB']
        },
        'clinicdb': {
            'name': 'CVC-ClinicDB',
            'type': '2D_to_3D',
            'classes': ['ClinicDB']
        },
        'btad': {
            'name': 'BTAD',
            'type': '2D_to_3D',
            'classes': ['01', '02', '03']
        }
    }

    @classmethod
    def get_supported_datasets(cls) -> Dict[str, Dict]:
        """获取支持的数据集列表"""
        return cls.SUPPORTED_DATASETS.copy()

    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        if dataset_name not in cls.SUPPORTED_DATASETS:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        return cls.SUPPORTED_DATASETS[dataset_name].copy()

    @classmethod
    def create_transforms(cls, image_size: int = 224,
                          normalize: bool = True) -> Dict[str, Any]:
        """创建标准的数据变换"""

        # 基础变换
        base_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]

        # 归一化变换
        if normalize:
            normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            base_transforms.append(normalize_transform)

        img_transform = transforms.Compose(base_transforms)

        # 深度图变换（不需要归一化）
        depth_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # 掩码变换
        mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        return {
            'img_transform': img_transform,
            'depth_transform': depth_transform,
            'mask_transform': mask_transform
        }


def get_dataset(dataset_name: str,
                split: str = 'train',
                classes: Optional[List[str]] = None,
                image_size: int = 224,
                normalize: bool = True,
                cache_depth: bool = True,
                preprocess_all: bool = False,
                **kwargs):
    """
    统一的数据集获取接口

    Args:
        dataset_name: 数据集名称
        split: 数据划分 ('train', 'test', 'val')
        classes: 类别列表，None表示使用所有类别
        image_size: 图像尺寸
        normalize: 是否归一化
        cache_depth: 是否缓存深度图
        preprocess_all: 是否预处理整个数据集
        **kwargs: 其他参数

    Returns:
        GeoCLIP数据集实例
    """

    # 检查数据集支持
    factory = DatasetFactory()
    if dataset_name not in factory.SUPPORTED_DATASETS:
        available = list(factory.SUPPORTED_DATASETS.keys())
        raise ValueError(f"不支持的数据集: {dataset_name}。支持的数据集: {available}")

    # 获取数据集信息
    dataset_info = factory.get_dataset_info(dataset_name)

    # 设置类别
    if classes is None:
        classes = dataset_info['classes']

    # 创建变换
    transforms_dict = factory.create_transforms(image_size, normalize)

    # 确定训练模式
    training = (split == 'train')

    # 创建数据集
    dataset = create_geoclip_dataset(
        dataset_name=dataset_name,
        clsnames=classes,
        transform=transforms_dict['img_transform'],
        target_transform=transforms_dict['mask_transform'],
        depth_transform=transforms_dict['depth_transform'],
        training=training,
        cache_depth=cache_depth,
        preprocess_all=preprocess_all,
        depth_cache_dir=kwargs.get('depth_cache_dir', f'./depth_cache/{dataset_name}'),
        **kwargs
    )

    return dataset


# 便利函数
def create_train_test_datasets(dataset_name: str,
                               classes: Optional[List[str]] = None,
                               image_size: int = 224,
                               cache_depth: bool = True,
                               preprocess_all: bool = False,
                               **kwargs):
    """
    同时创建训练和测试数据集

    Returns:
        tuple: (train_dataset, test_dataset)
    """

    train_dataset = get_dataset(
        dataset_name=dataset_name,
        split='train',
        classes=classes,
        image_size=image_size,
        cache_depth=cache_depth,
        preprocess_all=preprocess_all,
        **kwargs
    )

    test_dataset = get_dataset(
        dataset_name=dataset_name,
        split='test',
        classes=classes,
        image_size=image_size,
        cache_depth=cache_depth,
        preprocess_all=False,  # 测试集通常不需要预处理
        **kwargs
    )

    return train_dataset, test_dataset


# 使用示例
if __name__ == "__main__":
    # 查看支持的数据集
    factory = DatasetFactory()
    print("支持的数据集:")
    for name, info in factory.get_supported_datasets().items():
        print(f"  {name}: {info['name']} ({len(info['classes'])} 类别)")

    # 创建数据集
    try:
        dataset = get_dataset(
            dataset_name='mvtec',
            split='train',
            classes=['bottle'],
            image_size=224,
            cache_depth=True
        )
        print(f"\n创建数据集成功: {len(dataset)} 个样本")

        # 测试样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本键: {list(sample.keys())}")
            print(f"图像形状: {sample['img'].shape}")
            print(f"深度形状: {sample['depth'].shape}")

    except Exception as e:
        print(f"数据集创建测试跳过: {e}")