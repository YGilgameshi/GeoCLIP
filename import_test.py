"""
GeoCLIP - 导入测试脚本
测试所有模块的导入是否正常
"""

import sys
from pathlib import Path


def test_imports():
    """测试所有GeoCLIP模块的导入"""
    print("🧪 GeoCLIP模块导入测试")
    print("=" * 50)

    # 测试结果记录
    results = {}

    # 1. 测试核心模型模块
    print("\n📦 测试核心模型模块:")

    try:
        from geoclip.models.depth_estimator import DepthEstimator, create_depth_estimator
        print("✅ depth_estimator - 深度估计器")
        results['depth_estimator'] = True
    except Exception as e:
        print(f"❌ depth_estimator - {e}")
        results['depth_estimator'] = False

    try:
        from geoclip.models.geometry_encoder import (
            VoxelEncoder, SparseVoxelEncoder, HierarchicalVoxelEncoder,
            create_geometry_encoder
        )
        print("✅ geometry_encoder - 几何编码器")
        results['geometry_encoder'] = True
    except Exception as e:
        print(f"❌ geometry_encoder - {e}")
        results['geometry_encoder'] = False

    try:
        from geoclip.models.fusion_module import (
            FeatureFusionModule, BilinearFusionModule, ResidualFusionModule,
            create_fusion_module
        )
        print("✅ fusion_module - 特征融合模块")
        results['fusion_module'] = True
    except Exception as e:
        print(f"❌ fusion_module - {e}")
        results['fusion_module'] = False

    try:
        from geoclip.models.geoclip_main import GeoCLIP, create_geoclip_model
        print("✅ geoclip - 主模型")
        results['geoclip'] = True
    except Exception as e:
        print(f"❌ geoclip - {e}")
        results['geoclip'] = False

    # 2. 测试训练模块
    print("\n🏋️ 测试训练模块:")

    try:
        from geoclip.training.losses import (
            ContrastiveLoss, GeometryConsistencyLoss, AnomalyDetectionLoss,
            GeoCLIPLoss, create_loss_function
        )
        print("✅ losses - 损失函数")
        results['losses'] = True
    except Exception as e:
        print(f"❌ losses - {e}")
        results['losses'] = False

    try:
        from geoclip.training.trainer import GeoCLIPTrainer, create_trainer
        print("✅ trainer - 训练器")
        results['trainer'] = True
    except Exception as e:
        print(f"❌ trainer - {e}")
        results['trainer'] = False

    # 3. 测试工具模块
    print("\n🛠️ 测试工具模块:")

    try:
        from geoclip.utils.voxel_utils import (
            depth_to_pointcloud, pointcloud_to_voxel, DepthToVoxelConverter
        )
        print("✅ voxel_utils - 体素转换工具")
        results['voxel_utils'] = True
    except Exception as e:
        print(f"❌ voxel_utils - {e}")
        results['voxel_utils'] = False

    try:
        from geoclip.utils.metrics import AnomalyMetrics, PerPixelMetrics
        print("✅ metrics - 评估指标")
        results['metrics'] = True
    except Exception as e:
        print(f"❌ metrics - {e}")
        results['metrics'] = False

    try:
        from geoclip.utils.visualization import Visualizer
        print("✅ visualization - 可视化工具")
        results['visualization'] = True
    except Exception as e:
        print(f"❌ visualization - {e}")
        results['visualization'] = False

    # 4. 测试数据集模块
    print("\n📊 测试数据集模块:")

    try:
        from geoclip.datasets.adaclip_adapter import (
            AdaCLIPToGeoCLIPAdapter, GeoCLIP_AdaCLIPDataset, create_geoclip_dataset
        )
        print("✅ adaclip_adapter - 数据集适配器")
        results['adaclip_adapter'] = True
    except Exception as e:
        print(f"❌ adaclip_adapter - {e}")
        results['adaclip_adapter'] = False

    # 5. 测试可选依赖
    print("\n🔌 测试可选依赖:")

    # wandb
    try:
        import wandb
        print("✅ wandb - 实验跟踪 (可选)")
        results['wandb'] = True
    except ImportError:
        print("⚠️ wandb - 未安装 (可选，用于实验跟踪)")
        results['wandb'] = False

    # sklearn
    try:
        from sklearn.metrics import roc_auc_score
        print("✅ scikit-learn - 机器学习工具")
        results['sklearn'] = True
    except ImportError:
        print("❌ scikit-learn - 未安装 (必需)")
        results['sklearn'] = False

    # matplotlib
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib - 绘图库")
        results['matplotlib'] = True
    except ImportError:
        print("❌ matplotlib - 未安装 (可视化必需)")
        results['matplotlib'] = False

    # seaborn
    try:
        import seaborn as sns
        print("✅ seaborn - 统计绘图")
        results['seaborn'] = True
    except ImportError:
        print("⚠️ seaborn - 未安装 (可视化增强，可选)")
        results['seaborn'] = False

    # opencv
    try:
        import cv2
        print("✅ opencv-python - 计算机视觉")
        results['opencv'] = True
    except ImportError:
        print("❌ opencv-python - 未安装 (图像处理必需)")
        results['opencv'] = False

    # 6. 统计结果
    print("\n📋 导入测试总结:")
    print("=" * 50)

    core_modules = ['depth_estimator', 'geometry_encoder', 'fusion_module', 'geoclip']
    training_modules = ['losses', 'trainer']
    utils_modules = ['voxel_utils', 'metrics', 'visualization']
    data_modules = ['adaclip_adapter']
    optional_modules = ['wandb', 'seaborn']
    required_deps = ['sklearn', 'matplotlib', 'opencv']

    # 核心模块
    core_success = sum(results.get(m, False) for m in core_modules)
    print(f"🎯 核心模块: {core_success}/{len(core_modules)} ({'✅' if core_success == len(core_modules) else '❌'})")

    # 训练模块
    training_success = sum(results.get(m, False) for m in training_modules)
    print(
        f"🏋️ 训练模块: {training_success}/{len(training_modules)} ({'✅' if training_success == len(training_modules) else '❌'})")

    # 工具模块
    utils_success = sum(results.get(m, False) for m in utils_modules)
    print(f"🛠️ 工具模块: {utils_success}/{len(utils_modules)} ({'✅' if utils_success == len(utils_modules) else '❌'})")

    # 数据模块
    data_success = sum(results.get(m, False) for m in data_modules)
    print(f"📊 数据模块: {data_success}/{len(data_modules)} ({'✅' if data_success == len(data_modules) else '❌'})")

    # 必需依赖
    deps_success = sum(results.get(m, False) for m in required_deps)
    print(f"📦 必需依赖: {deps_success}/{len(required_deps)} ({'✅' if deps_success == len(required_deps) else '❌'})")

    # 可选依赖
    optional_success = sum(results.get(m, False) for m in optional_modules)
    print(
        f"🔌 可选依赖: {optional_success}/{len(optional_modules)} ({'✅' if optional_success == len(optional_modules) else '⚠️'})")

    # 总体评估
    total_required = len(core_modules) + len(training_modules) + len(utils_modules) + len(data_modules) + len(
        required_deps)
    total_success = core_success + training_success + utils_success + data_success + deps_success

    print(
        f"\n🎉 总体评估: {total_success}/{total_required} ({'✅ 可以正常使用' if total_success == total_required else '❌ 需要修复问题'})")

    # 给出建议
    print("\n💡 建议:")
    if total_success == total_required:
        print("✅ 所有必需模块都可以正常导入，项目可以正常运行！")
    else:
        print("❌ 存在导入问题，请按以下步骤修复：")

        # 检查缺失的必需依赖
        missing_deps = [m for m in required_deps if not results.get(m, False)]
        if missing_deps:
            print(f"\n1. 安装缺失的必需依赖:")
            for dep in missing_deps:
                if dep == 'sklearn':
                    print(f"   pip install scikit-learn")
                elif dep == 'matplotlib':
                    print(f"   pip install matplotlib")
                elif dep == 'opencv':
                    print(f"   pip install opencv-python")

        # 检查模块导入问题
        failed_modules = [m for m in core_modules + training_modules + utils_modules + data_modules
                          if not results.get(m, False)]
        if failed_modules:
            print(f"\n2. 修复模块导入问题:")
            print(f"   检查以下模块: {', '.join(failed_modules)}")
            print(f"   确保GeoCLIP项目路径正确，所有.py文件存在")

        # 可选依赖建议
        if not results.get('wandb', False):
            print(f"\n3. 可选：安装wandb用于实验跟踪:")
            print(f"   pip install wandb")

        if not results.get('seaborn', False):
            print(f"\n4. 可选：安装seaborn增强可视化:")
            print(f"   pip install seaborn")

    return results


def test_basic_functionality():
    """测试基础功能是否正常"""
    print("\n🧪 测试基础功能")
    print("=" * 30)

    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")

        # 测试创建简单模型
        from geoclip.models.fusion_module import create_fusion_module

        config = {
            'type': 'simple_concat',
            'clip_dim': 512,
            'geometry_dim': 512,
            'output_dim': 512
        }

        fusion_module = create_fusion_module(config)
        print("✅ 融合模块创建成功")

        # 测试前向传播
        clip_feat = torch.randn(2, 512)
        geometry_feat = torch.randn(2, 512)

        with torch.no_grad():
            output = fusion_module(clip_feat, geometry_feat)

        print(f"✅ 前向传播成功: {clip_feat.shape} + {geometry_feat.shape} -> {output.shape}")

        # 测试指标计算
        from geoclip.utils.metrics import AnomalyMetrics

        metrics = AnomalyMetrics()
        pred = torch.rand(100)
        labels = torch.randint(0, 2, (100,))

        result = metrics.compute_metrics(pred.numpy(), labels.numpy())
        print(f"✅ 指标计算成功: AUC = {result.get('auc', 0):.3f}")

        print("\n🎉 基础功能测试通过！")
        return True

    except Exception as e:
        print(f"\n❌ 基础功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 GeoCLIP项目完整性检查")
    print("=" * 60)

    # 1. 导入测试
    import_results = test_imports()

    # 2. 基础功能测试
    if all(import_results.get(m, False) for m in ['fusion_module', 'metrics']):
        functionality_ok = test_basic_functionality()
    else:
        print("\n⚠️ 跳过功能测试 - 基础模块导入失败")
        functionality_ok = False

    # 3. 最终评估
    print("\n" + "=" * 60)
    print("🏁 最终评估")

    core_modules = ['depth_estimator', 'geometry_encoder', 'fusion_module', 'geoclip']
    core_ok = all(import_results.get(m, False) for m in core_modules)

    if core_ok and functionality_ok:
        print("🎉 恭喜！GeoCLIP项目已准备就绪，可以开始使用！")
        print("\n📚 接下来可以:")
        print("1. 运行 geoclip_usage_example.py 查看完整使用示例")
        print("2. 使用 GeoCLIP 进行异常检测实验")
        print("3. 根据需要调整模型配置和超参数")
    else:
        print("❌ 项目还需要进一步配置才能正常使用")
        print("请根据上面的建议修复问题后重新运行此脚本")

    return core_ok and functionality_ok


if __name__ == "__main__":
    success = main()