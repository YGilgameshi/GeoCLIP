"""
GeoCLIP - AdaCLIP数据集适配器
将原有的2D数据集适配为3D数据集，通过深度估计生成伪3D数据
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
from tqdm import tqdm
from PIL import Image
import hashlib

# 导入AdaCLIP原有的数据集基类
try:
    from dataset.base_dataset import BaseDataset
except ImportError:
    print("警告: 无法导入BaseDataset，请确保在AdaCLIP项目根目录下运行")

from geoclip.models.depth_estimator import DepthEstimator
from geoclip.utils.transforms import GeoCLIPTransforms


class AdaCLIPToGeoCLIPAdapter:
    """
    AdaCLIP数据集到GeoCLIP的适配器
    为原有2D数据集生成深度信息
    """

    def __init__(self,
                 depth_estimator: Optional[DepthEstimator] = None,
                 cache_depth: bool = True,
                 depth_cache_dir: str = "./depth_cache",
                 device: str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化深度估计器
        if depth_estimator is None:
            print("初始化深度估计器...")
            self.depth_estimator = DepthEstimator(
                model_type="DPT_Large",  # 使用修复后的模型名称
                device=self.device
            )
        else:
            self.depth_estimator = depth_estimator

        self.cache_depth = cache_depth
        self.depth_cache_dir = Path(depth_cache_dir)

        if self.cache_depth:
            self.depth_cache_dir.mkdir(exist_ok=True, parents=True)
            print(f"深度缓存目录: {self.depth_cache_dir}")

        self.transforms = GeoCLIPTransforms()

        print(f"AdaCLIP数据集适配器初始化完成")
        print(f"深度缓存: {'启用' if cache_depth else '禁用'}")
        print(f"运行设备: {self.device}")

    def get_depth_cache_path(self, image_path: str) -> str:
        """获取深度缓存文件路径"""
        # 使用MD5 hash生成稳定的缓存文件名
        path_str = str(image_path)
        path_hash = hashlib.md5(path_str.encode()).hexdigest()[:16]
        cache_filename = f"depth_{path_hash}.npy"
        return str(self.depth_cache_dir / cache_filename)

    def estimate_or_load_depth(self, image_path: str, image: np.ndarray = None) -> np.ndarray:
        """估计或加载缓存的深度图"""
        cache_path = self.get_depth_cache_path(image_path)

        # 尝试从缓存加载
        if self.cache_depth and os.path.exists(cache_path):
            try:
                depth = np.load(cache_path)
                return depth.astype(np.float32)
            except Exception as e:
                print(f"加载缓存失败 {cache_path}: {e}")

        # 如果没有提供图像，从路径加载
        if image is None:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 估计深度
        depth = self.estimate_depth(image)

        # 保存到缓存
        if self.cache_depth:
            try:
                np.save(cache_path, depth)
            except Exception as e:
                print(f"保存缓存失败 {cache_path}: {e}")

        return depth

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """使用深度估计器估计深度"""
        try:
            # 预处理图像
            if len(image.shape) == 3:
                # RGB图像
                if image.dtype == np.uint8:
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                else:
                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                # 灰度图像
                if image.dtype == np.uint8:
                    image_tensor = torch.from_numpy(image).float() / 255.0
                else:
                    image_tensor = torch.from_numpy(image).float()

                # 扩展到3通道
                if len(image_tensor.shape) == 2:
                    image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)

            # 添加batch维度
            image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # 深度估计
            with torch.no_grad():
                depth_tensor = self.depth_estimator(image_tensor)

            # 转换回numpy
            if depth_tensor.dim() == 4:
                depth = depth_tensor[0, 0].cpu().numpy()  # [H, W]
            elif depth_tensor.dim() == 3:
                depth = depth_tensor[0].cpu().numpy()
            else:
                depth = depth_tensor.cpu().numpy()

            return depth.astype(np.float32)

        except Exception as e:
            print(f"深度估计失败: {e}")
            # 返回零深度图作为fallback
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.float32)

    def preprocess_dataset(self, dataset_class, dataset_config: Dict[str, Any]) -> Dict[str, int]:
        """
        预处理整个数据集，生成所有深度图

        Args:
            dataset_class: 数据集类
            dataset_config: 数据集配置参数

        Returns:
            处理统计信息
        """

        print(f"开始预处理数据集: {dataset_config.get('name', 'Unknown')}")

        stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'new_estimates': 0,
            'errors': 0
        }

        # 处理训练和测试数据
        for training in [True, False]:
            phase = 'train' if training else 'test'
            print(f"处理 {phase} 数据...")

            try:
                # 创建数据集实例（使用恒等变换避免复杂处理）
                config = dataset_config.copy()
                config['training'] = training
                config['transform'] = lambda x: x
                config['target_transform'] = lambda x: x

                dataset = dataset_class(**config)

                # 遍历数据集
                for idx in tqdm(range(len(dataset)), desc=f"Processing {phase}"):
                    try:
                        # 获取样本信息（但不实际加载数据）
                        data_info = dataset.data_all[idx]
                        img_path = data_info['img_path']
                        full_path = os.path.join(dataset.root, img_path)

                        # 检查缓存
                        cache_path = self.get_depth_cache_path(full_path)
                        if os.path.exists(cache_path):
                            stats['cache_hits'] += 1
                        else:
                            # 估计深度（会自动缓存）
                            _ = self.estimate_or_load_depth(full_path)
                            stats['new_estimates'] += 1

                        stats['total_processed'] += 1

                    except Exception as e:
                        print(f"处理样本 {idx} 时出错: {e}")
                        stats['errors'] += 1

            except Exception as e:
                print(f"创建 {phase} 数据集时出错: {e}")

        print(f"数据集预处理完成:")
        print(f"  总处理: {stats['total_processed']}")
        print(f"  缓存命中: {stats['cache_hits']}")
        print(f"  新估计: {stats['new_estimates']}")
        print(f"  错误: {stats['errors']}")

        return stats


class GeoCLIP_AdaCLIPDataset(BaseDataset):
    """
    GeoCLIP版本的AdaCLIP数据集
    在原有数据集基础上添加深度信息
    """

    def __init__(self,
                 dataset_name: str,
                 clsnames: List[str],
                 transform,
                 target_transform,
                 root: str,
                 aug_rate: float = 0.0,
                 training: bool = True,
                 depth_adapter: Optional[AdaCLIPToGeoCLIPAdapter] = None,
                 depth_transform=None):

        self.dataset_name = dataset_name
        self.depth_adapter = depth_adapter or AdaCLIPToGeoCLIPAdapter()
        self.depth_transform = depth_transform

        # 调用父类初始化
        super().__init__(
            clsnames=clsnames,
            transform=transform,
            target_transform=target_transform,
            root=root,
            aug_rate=aug_rate,
            training=training
        )

        print(f"GeoCLIP-{dataset_name} 数据集初始化完成")
        print(f"  样本数量: {len(self.data_all)}")
        print(f"  类别数量: {len(self.cls_names)}")

    def __getitem__(self, index):
        """重写getitem方法，添加深度信息"""
        # 获取原有数据
        original_sample = super().__getitem__(index)

        # 添加深度信息
        img_path = self.data_all[index]['img_path']
        full_img_path = os.path.join(self.root, img_path)

        try:
            # 估计或加载深度
            depth = self.depth_adapter.estimate_or_load_depth(full_img_path)

            # 应用深度变换
            if self.depth_transform is not None:
                # 归一化深度图到[0, 1]范围
                depth_min, depth_max = depth.min(), depth.max()
                if depth_max > depth_min:
                    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
                else:
                    depth_normalized = np.zeros_like(depth)

                # 转换为PIL Image
                depth_pil = Image.fromarray((depth_normalized * 255).astype(np.uint8), mode='L')
                depth_transformed = self.depth_transform(depth_pil)

                # 如果是tensor，转换回原始深度值范围
                if isinstance(depth_transformed, torch.Tensor):
                    if depth_max > depth_min:
                        depth_final = depth_transformed.float() * (depth_max - depth_min) + depth_min
                    else:
                        depth_final = depth_transformed.float()
                else:
                    depth_final = np.array(depth_transformed, dtype=np.float32)
                    if depth_max > depth_min:
                        depth_final = depth_final * (depth_max - depth_min) + depth_min

            else:
                # 直接转换为tensor
                depth_final = torch.from_numpy(depth).float()
                if depth_final.dim() == 2:
                    depth_final = depth_final.unsqueeze(0)  # 添加通道维度

            # 添加到样本中
            original_sample['depth'] = depth_final
            original_sample['has_depth'] = True
            original_sample['depth_range'] = (depth.min().item(), depth.max().item())

        except Exception as e:
            print(f"为样本 {index} 生成深度图时出错: {e}")
            # 创建零深度图作为fallback
            if hasattr(original_sample['img'], 'shape'):
                h, w = original_sample['img'].shape[-2:]
                original_sample['depth'] = torch.zeros(1, h, w, dtype=torch.float32)
            else:
                original_sample['depth'] = torch.zeros(1, 256, 256, dtype=torch.float32)
            original_sample['has_depth'] = False
            original_sample['depth_range'] = (0.0, 0.0)

        return original_sample


# 便利的工厂函数
def create_geoclip_dataset(dataset_name: str,
                           clsnames: List[str] = None,
                           transform=None,
                           target_transform=None,
                           depth_transform=None,
                           root: str = None,
                           training: bool = True,
                           preprocess_all: bool = False,
                           cache_depth: bool = True,
                           depth_cache_dir: str = None,
                           **kwargs) -> GeoCLIP_AdaCLIPDataset:
    """
    创建GeoCLIP版本的AdaCLIP数据集

    Args:
        dataset_name: 数据集名称 ('mvtec', 'visa', 'colondb', 等)
        clsnames: 类别名称列表
        transform: 图像变换
        target_transform: 目标变换
        depth_transform: 深度图变换
        root: 数据集根目录
        training: 是否为训练模式
        preprocess_all: 是否预处理整个数据集
        cache_depth: 是否缓存深度图
        depth_cache_dir: 深度缓存目录
        **kwargs: 其他参数

    Returns:
        GeoCLIP数据集实例
    """

    # 导入对应的数据集信息
    dataset_class = None
    default_clsnames = None
    default_root = None

    if dataset_name == 'mvtec':
        try:
            from dataset.mvtec import MVTecDataset, MVTEC_CLS_NAMES, MVTEC_ROOT
            dataset_class = MVTecDataset
            default_clsnames = MVTEC_CLS_NAMES
            default_root = MVTEC_ROOT
        except ImportError:
            raise ImportError(f"无法导入{dataset_name}数据集，请检查导入路径")

    elif dataset_name == 'visa':
        try:
            from dataset.visa import VisaDataset, VISA_CLS_NAMES, VISA_ROOT
            dataset_class = VisaDataset
            default_clsnames = VISA_CLS_NAMES
            default_root = VISA_ROOT
        except ImportError:
            raise ImportError(f"无法导入{dataset_name}数据集，请检查导入路径")

    elif dataset_name == 'colondb':
        try:
            from dataset.colondb import ColonDBDataset, ColonDB_CLS_NAMES, ColonDB_ROOT
            dataset_class = ColonDBDataset
            default_clsnames = ColonDB_CLS_NAMES
            default_root = ColonDB_ROOT
        except ImportError:
            raise ImportError(f"无法导入{dataset_name}数据集，请检查导入路径")

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 使用默认值
    if clsnames is None:
        clsnames = default_clsnames
    if root is None:
        root = default_root
    if depth_cache_dir is None:
        depth_cache_dir = f"./depth_cache_{dataset_name}"

    # 创建适配器
    adapter = AdaCLIPToGeoCLIPAdapter(
        cache_depth=cache_depth,
        depth_cache_dir=depth_cache_dir
    )

    # 如果需要预处理整个数据集
    if preprocess_all:
        print("开始预处理数据集...")
        dataset_config = {
            'name': dataset_name,
            'clsnames': clsnames,
            'root': root,
            **kwargs
        }
        adapter.preprocess_dataset(dataset_class, dataset_config)

    # 创建数据集
    dataset = GeoCLIP_AdaCLIPDataset(
        dataset_name=dataset_name,
        clsnames=clsnames,
        transform=transform,
        target_transform=target_transform,
        depth_transform=depth_transform,
        root=root,
        training=training,
        depth_adapter=adapter,
        **{k: v for k, v in kwargs.items()
           if k not in ['cache_depth', 'depth_cache_dir']}
    )

    return dataset


# 测试和使用示例
def test_adapter():
    """测试适配器功能"""
    print("=== 测试AdaCLIP数据集适配器 ===")

    try:
        # 1. 测试深度估计适配器
        print("\n1. 测试深度估计适配器")
        adapter = AdaCLIPToGeoCLIPAdapter(
            cache_depth=True,
            depth_cache_dir="./test_depth_cache"
        )

        # 创建测试图像
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        test_path = "test_image.jpg"

        # 测试深度估计
        depth = adapter.estimate_or_load_depth(test_path, test_image)
        print(f"✅ 深度估计成功:")
        print(f"   深度形状: {depth.shape}")
        print(f"   深度范围: {depth.min():.3f} - {depth.max():.3f}")

        # 测试缓存功能
        depth2 = adapter.estimate_or_load_depth(test_path, test_image)
        cache_works = np.allclose(depth, depth2)
        print(f"✅ 缓存测试: {'成功' if cache_works else '失败'}")

        # 2. 测试数据集创建（可能需要实际数据集）
        print("\n2. 测试数据集创建")
        try:
            import torchvision.transforms as transforms

            # 简单的transform
            simple_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            depth_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            dataset = create_geoclip_dataset(
                dataset_name='mvtec',
                clsnames=['bottle'],  # 只测试一个类别
                transform=simple_transform,
                target_transform=simple_transform,
                depth_transform=depth_transform,
                training=True,
                preprocess_all=False,
                cache_depth=True
            )

            print(f"✅ GeoCLIP数据集创建成功: {len(dataset)} 个样本")

            # 测试获取样本
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ 样本键: {list(sample.keys())}")
                if 'depth' in sample:
                    print(f"   深度形状: {sample['depth'].shape}")
                    print(f"   是否有深度: {sample['has_depth']}")
                    print(f"   深度范围: {sample.get('depth_range', 'N/A')}")

        except Exception as e:
            print(f"⚠️ 数据集测试跳过 (可能缺少数据): {e}")

        print("\n🎉 适配器测试完成!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_adapter()