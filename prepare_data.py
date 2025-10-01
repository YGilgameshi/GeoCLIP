"""
GeoCLIP 数据准备和验证脚本
基于AdaCLIP的数据集结构，准备GeoCLIP训练所需的数据
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

# 导入配置和GeoCLIP组件
from config import (
    check_dataset_availability, get_dataset_info, auto_detect_best_config,
    DATA_ROOT, DEPTH_CACHE_ROOT
)
from geoclip.datasets.adaclip_adapter import create_geoclip_dataset, AdaCLIPToGeoCLIPAdapter
from geoclip.models.depth_estimator import DepthEstimator


def validate_dataset_structure(dataset_name: str):
    """验证数据集结构是否符合AdaCLIP格式"""
    print(f"🔍 验证数据集结构: {dataset_name}")

    info = get_dataset_info(dataset_name)
    if not info:
        print(f"❌ 不支持的数据集: {dataset_name}")
        return False

    root_path = info['root']
    if not root_path.exists():
        print(f"❌ 数据集路径不存在: {root_path}")
        return False

    # 检查基本目录结构
    expected_dirs = []

    if dataset_name == 'mvtec':
        # MVTec AD 结构检查
        for cls_name in info['classes'][:3]:  # 检查前3个类别作为示例
            train_dir = root_path / cls_name / 'train'
            test_dir = root_path / cls_name / 'test'
            if train_dir.exists() and test_dir.exists():
                expected_dirs.append(cls_name)

    elif dataset_name in ['colondb', 'clinicdb']:
        # 医学数据集结构检查
        if (root_path / 'train').exists() and (root_path / 'test').exists():
            expected_dirs = ['train', 'test']

    if expected_dirs:
        print(f"✅ 数据集结构正确，发现目录: {expected_dirs}")
        return True
    else:
        print(f"❌ 数据集结构不完整")
        return False


def test_dataset_loading(dataset_name: str, max_samples: int = 5):
    """测试数据集加载功能"""
    print(f"🧪 测试数据集加载: {dataset_name}")

    try:
        # 创建简单的transform
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # 创建数据集（不预处理深度图）
        dataset = create_geoclip_dataset(
            dataset_name=dataset_name,
            clsnames=None,
            transform=transform,
            target_transform=transform,
            training=True,
            cache_depth=False,  # 暂时不缓存，快速测试
            preprocess_all=False
        )

        print(f"✅ 数据集加载成功: {len(dataset)} 个样本")

        # 测试读取样本
        if len(dataset) > 0:
            print(f"🔬 测试前 {min(max_samples, len(dataset))} 个样本...")

            for i in range(min(max_samples, len(dataset))):
                try:
                    sample = dataset[i]
                    print(f"   样本 {i}: keys={list(sample.keys())}")

                    if 'img' in sample:
                        print(f"     图像形状: {sample['img'].shape}")
                    if 'depth' in sample:
                        print(f"     深度形状: {sample['depth'].shape}")
                        print(f"     深度范围: {sample.get('depth_range', 'N/A')}")
                    if 'anomaly' in sample:
                        print(f"     异常标签: {sample['anomaly']}")

                except Exception as e:
                    print(f"     ❌ 样本 {i} 加载失败: {e}")
                    return False

            print(f"✅ 样本测试完成")
            return True
        else:
            print(f"❌ 数据集为空")
            return False

    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False


def preprocess_depth_cache(dataset_name: str, max_samples: int = None, force: bool = False):
    """预处理深度缓存"""
    print(f"⚙️ 预处理深度缓存: {dataset_name}")

    # 创建深度缓存目录
    cache_dir = DEPTH_CACHE_ROOT / dataset_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已有缓存
    existing_cache = list(cache_dir.glob("depth_*.npy"))
    if existing_cache and not force:
        print(f"ℹ️ 已存在 {len(existing_cache)} 个深度缓存文件")
        user_input = input("是否重新生成？(y/N): ")
        if user_input.lower() != 'y':
            print("跳过深度预处理")
            return True

    try:
        # 创建适配器
        adapter = AdaCLIPToGeoCLIPAdapter(
            cache_depth=True,
            depth_cache_dir=str(cache_dir)
        )

        print(f"✅ 深度估计器初始化完成")

        # 创建数据集进行预处理
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # 分别处理训练和测试数据
        total_processed = 0

        for training in [True, False]:
            phase = "训练" if training else "测试"
            print(f"🔄 处理{phase}数据...")

            try:
                dataset = create_geoclip_dataset(
                    dataset_name=dataset_name,
                    clsnames=None,
                    transform=transform,
                    target_transform=transform,
                    training=training,
                    cache_depth=True,
                    preprocess_all=False,
                    depth_adapter=adapter
                )

                # 限制处理样本数量
                num_samples = len(dataset)
                if max_samples and num_samples > max_samples:
                    num_samples = max_samples
                    print(f"⚠️ 限制处理样本数: {num_samples}")

                # 逐个处理样本生成深度缓存
                for i in tqdm(range(num_samples), desc=f"处理{phase}样本"):
                    try:
                        # 获取样本会自动生成深度缓存
                        _ = dataset[i]
                        total_processed += 1
                    except Exception as e:
                        print(f"❌ 处理样本 {i} 失败: {e}")
                        continue

            except Exception as e:
                print(f"❌ 处理{phase}数据失败: {e}")
                continue

        print(f"✅ 深度预处理完成，总共处理 {total_processed} 个样本")

        # 统计生成的缓存文件
        final_cache = list(cache_dir.glob("depth_*.npy"))
        print(f"📁 生成深度缓存文件: {len(final_cache)} 个")

        return True

    except Exception as e:
        print(f"❌ 深度预处理失败: {e}")
        return False


def validate_geoclip_integration():
    """验证GeoCLIP组件集成"""
    print("🔗 验证GeoCLIP组件集成...")

    try:
        # # 测试深度估计器
        # print("  测试深度估计器...")
        # depth_estimator = DepthEstimator()
        # test_image = torch.randn(1, 3, 224, 224)
        # with torch.no_grad():
        #     depth_output = depth_estimator(test_image)
        # print(f"    ✅ 深度估计器: {test_image.shape} -> {depth_output.shape}")

        # 测试GeoCLIP主模型
        print("  测试GeoCLIP主模型...")
        from geoclip.models.geoclip_main import create_geoclip_model

        model_config = {
            'clip_model': 'ViT-B-16',
            'detection_type': 'regression',
            'device': 'cpu'  # 使用CPU进行测试
        }

        model = create_geoclip_model(model_config)
        test_batch = {'image': torch.randn(1, 3, 224, 224)}

        with torch.no_grad():
            results = model(test_batch)

        print(f"    ✅ GeoCLIP模型测试成功")
        for key, value in results.items():
            if torch.is_tensor(value):
                print(f"      {key}: {value.shape}")

        return True

    except Exception as e:
        print(f"    ❌ 组件集成测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser("GeoCLIP数据准备工具")

    parser.add_argument("--action", type=str,
                        choices=['check', 'validate', 'test_load', 'preprocess', 'full_check'],
                        default='full_check',
                        help="执行的操作")

    parser.add_argument("--dataset", type=str,
                        choices=['mvtec', 'visa', 'btad', 'colondb', 'clinicdb', 'all'],
                        default='all',
                        help="操作的数据集")

    parser.add_argument("--max_samples", type=int, default=100,
                        help="预处理或测试的最大样本数")

    parser.add_argument("--force", action='store_true',
                        help="强制重新处理")

    args = parser.parse_args()

    print("🚀 GeoCLIP数据准备工具")
    print("=" * 40)

    if args.action == 'check':
        # 检查数据集可用性
        available, missing = check_dataset_availability()
        if missing:
            print(f"\n💡 建议：请先下载缺失的数据集: {missing}")

    elif args.action == 'validate':
        # 验证数据集结构
        datasets = [args.dataset] if args.dataset != 'all' else ['mvtec', 'visa', 'colondb']

        for dataset in datasets:
            if not validate_dataset_structure(dataset):
                print(f"❌ {dataset} 验证失败")
                return 1

    elif args.action == 'test_load':
        # 测试数据集加载
        datasets = [args.dataset] if args.dataset != 'all' else ['mvtec', 'visa', 'colondb']

        for dataset in datasets:
            if not test_dataset_loading(dataset, args.max_samples):
                print(f"❌ {dataset} 加载测试失败")
                return 1

    elif args.action == 'preprocess':
        # 预处理深度缓存
        datasets = [args.dataset] if args.dataset != 'all' else ['mvtec']  # 默认只处理mvtec

        for dataset in datasets:
            if not preprocess_depth_cache(dataset, args.max_samples, args.force):
                print(f"❌ {dataset} 深度预处理失败")
                return 1

    elif args.action == 'full_check':
        # 完整检查
        print("🔍 执行完整检查...")

        # 1. 检查数据集可用性
        available, missing = check_dataset_availability()

        # 2. 验证GeoCLIP集成
        if not validate_geoclip_integration():
            return 1

        # 3. 测试数据集加载
        test_datasets = ['mvtec'] if 'mvtec' in available else available[:1]
        for dataset in test_datasets:
            if not test_dataset_loading(dataset, 3):  # 只测试3个样本
                print(f"❌ {dataset} 测试失败")
                return 1

        # 4. 给出建议
        print("\n💡 准备建议:")
        best_config = auto_detect_best_config()
        if best_config:
            print("✅ 您的环境已准备就绪！")
            print("推荐使用以下命令开始训练:")
            print(f"  bash quick_train.sh standard")
        else:
            print("❌ 环境未准备就绪，请先下载数据集")

    print("\n🎉 数据准备检查完成！")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)