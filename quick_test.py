
"""
GeoCLIP 快速功能测试
使用最小配置快速验证核心功能是否正常
"""

import torch
import torch.nn as nn
import numpy as np
import time


def quick_test():
    """快速测试GeoCLIP核心功能"""
    print("⚡ GeoCLIP 快速功能测试")
    print("=" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ 设备: {device}")

    # 测试1: 融合模块
    print("\n1️⃣ 测试特征融合模块...")
    try:
        from geoclip.models.fusion_module import create_fusion_module

        fusion_config = {
            'type': 'simple_concat',
            'clip_dim': 256,
            'geometry_dim': 256,
            'output_dim': 256
        }

        fusion_module = create_fusion_module(fusion_config).to(device)

        # 测试数据
        clip_feat = torch.randn(2, 256, device=device)
        geometry_feat = torch.randn(2, 256, device=device)

        with torch.no_grad():
            output = fusion_module(clip_feat, geometry_feat)

        print(f"✅ 融合模块: {clip_feat.shape} + {geometry_feat.shape} -> {output.shape}")

    except Exception as e:
        print(f"❌ 融合模块失败: {e}")
        return False

    # 测试2: 体素转换
    print("\n2️⃣ 测试体素转换...")
    try:
        from geoclip.utils.voxel_utils import DepthToVoxelConverter

        converter = DepthToVoxelConverter(voxel_size=16, use_color=True)

        # 创建RGBD测试数据
        rgbd = torch.randn(2, 4, 64, 64, device=device)  # 小尺寸快速测试

        with torch.no_grad():
            voxels = converter.images_to_voxels(rgbd)

        print(f"✅ 体素转换: {rgbd.shape} -> {voxels.shape}")

    except Exception as e:
        print(f"❌ 体素转换失败: {e}")
        return False

    # 测试3: 几何编码器
    print("\n3️⃣ 测试几何编码器...")
    try:
        from geoclip.models.geometry_encoder import create_geometry_encoder

        config = {
            'type': 'voxel',
            'in_channels': 4,
            'base_channels': 16,  # 小通道数快速测试
            'num_stages': 2,
            'output_channels': 128,
            'voxel_size': 16
        }

        encoder = create_geometry_encoder(config).to(device)

        # 使用之前的体素数据
        with torch.no_grad():
            features = encoder(voxels)

        print(f"✅ 几何编码: {voxels.shape} -> {features.shape}")

    except Exception as e:
        print(f"❌ 几何编码失败: {e}")
        return False

    # 测试4: 指标计算
    print("\n4️⃣ 测试指标计算...")
    try:
        from geoclip.utils.metrics import AnomalyMetrics

        metrics = AnomalyMetrics()

        # 模拟预测和标签
        pred = np.random.rand(50)
        labels = np.random.randint(0, 2, 50)

        result = metrics.compute_metrics(pred, labels)

        print(f"✅ 指标计算: AUC={result.get('auc', 0):.3f}, F1={result.get('f1', 0):.3f}")

    except Exception as e:
        print(f"❌ 指标计算失败: {e}")
        return False

    # 测试5: 可视化
    print("\n5️⃣ 测试可视化...")
    try:
        from geoclip.utils.visualization import Visualizer

        viz = Visualizer(save_dir='geoclip/quick_test_viz')

        # 创建简单测试图
        test_img = np.random.rand(64, 64, 3)
        test_depth = np.random.rand(64, 64)

        viz.plot_depth_visualization(test_img, test_depth, 'quick_test_depth.png')

        print(f"✅ 可视化: 图片保存到 {viz.save_dir}")

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        return False

    # 测试6: 损失函数
    print("\n6️⃣ 测试损失函数...")
    try:
        from geoclip.training.losses import create_loss_function

        loss_config = {
            'type': 'geoclip',
            'contrastive_weight': 1.0,
            'geometry_weight': 0.5,
            'anomaly_weight': 2.0,
            'anomaly_config': {
                'loss_type': 'mse'  # 使用MSE损失避免分类问题
            }
        }

        loss_fn = create_loss_function(loss_config)

        # 模拟模型输出和目标 - 确保维度匹配
        clip_dim = 256
        geometry_dim = 128  # 不同维度测试自动匹配功能

        model_outputs = {
            'anomaly_predictions': torch.randn(2, 1, device=device).sigmoid(),
            'clip_features': torch.randn(2, clip_dim, device=device),
            'geometry_features': torch.randn(2, geometry_dim, device=device),
            'depth_maps': torch.rand(2, 1, 32, 32, device=device)
        }

        targets = {
            'anomaly_labels': torch.randint(0, 2, (2,), device=device).float(),  # 转为float用于回归
            'class_labels': torch.randint(0, 2, (2,), device=device)  # 二分类
        }

        with torch.no_grad():
            loss_dict = loss_fn(model_outputs, targets)

        print(f"✅ 损失函数: total_loss={loss_dict['total_loss']:.3f}")
        print(f"   anomaly_loss={loss_dict.get('anomaly_loss', 0):.3f}")
        print(f"   contrastive_loss={loss_dict.get('contrastive_loss', 0):.3f}")
        print(f"   geometry_loss={loss_dict.get('geometry_loss', 0):.3f}")

    except Exception as e:
        print(f"❌ 损失函数失败: {e}")
        return False

    # 测试7: 端到端流程
    print("\n7️⃣ 测试端到端流程...")
    try:
        # 创建简化的端到端测试
        batch_size = 2

        # 1. 模拟图像
        images = torch.randn(batch_size, 3, 128, 128, device=device)

        # 2. 模拟深度 (跳过实际深度估计以加速)
        depths = torch.rand(batch_size, 1, 128, 128, device=device)

        # 3. 体素转换
        rgbd = torch.cat([images, depths], dim=1)
        voxels = converter.images_to_voxels(rgbd)

        # 4. 特征提取
        geometry_feat = encoder(voxels)
        clip_feat = torch.randn(batch_size, 256, device=device)  # 模拟CLIP特征

        # 5. 特征融合
        fusion_config_e2e = {
            'type': 'simple_concat',
            'clip_dim': 256,
            'geometry_dim': geometry_feat.shape[1],
            'output_dim': 256
        }
        fusion_e2e = create_fusion_module(fusion_config_e2e).to(device)
        fused_feat = fusion_e2e(clip_feat, geometry_feat)

        # 6. 异常检测
        anomaly_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device)

        anomaly_scores = anomaly_head(fused_feat)

        print(f"✅ 端到端流程:")
        print(f"   图像: {images.shape}")
        print(f"   深度: {depths.shape}")
        print(f"   体素: {voxels.shape}")
        print(f"   融合特征: {fused_feat.shape}")
        print(f"   异常分数: {anomaly_scores.shape}")

    except Exception as e:
        print(f"❌ 端到端流程失败: {e}")
        return False

    return True


def performance_test():
    """性能测试"""
    print("\n⏱️ 性能测试...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_sizes = [1, 2, 4] if device == 'cuda' else [1, 2]

    try:
        from geoclip.models.fusion_module import create_fusion_module

        fusion_config = {
            'type': 'cross_attention',
            'clip_dim': 512,
            'geometry_dim': 512,
            'output_dim': 512
        }

        fusion_module = create_fusion_module(fusion_config).to(device)

        print("批次大小 | 处理时间(ms) | 内存使用(MB)")
        print("-" * 40)

        for batch_size in batch_sizes:
            clip_feat = torch.randn(batch_size, 512, device=device)
            geometry_feat = torch.randn(batch_size, 512, device=device)

            # 预热
            with torch.no_grad():
                _ = fusion_module(clip_feat, geometry_feat)

            if device == 'cuda':
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated() / 1024 / 1024

            # 测试时间
            start_time = time.time()

            with torch.no_grad():
                for _ in range(10):  # 多次运行取平均
                    output = fusion_module(clip_feat, geometry_feat)

            if device == 'cuda':
                torch.cuda.synchronize()

            elapsed_time = (time.time() - start_time) / 10 * 1000  # 转换为毫秒

            if device == 'cuda':
                end_memory = torch.cuda.memory_allocated() / 1024 / 1024
                memory_used = end_memory - start_memory
            else:
                memory_used = 0

            print(f"{batch_size:8d} | {elapsed_time:10.1f} | {memory_used:10.1f}")

        print("✅ 性能测试完成")

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")


def main():
    """主测试函数"""
    print("🚀 开始GeoCLIP快速测试...")

    start_time = time.time()

    # 功能测试
    success = quick_test()

    if success:
        print(f"\n🎉 所有功能测试通过!")

        # 性能测试
        performance_test()

        total_time = time.time() - start_time
        print(f"\n⏱️ 总测试时间: {total_time:.1f}秒")

        print(f"\n✅ GeoCLIP项目已准备就绪!")
        print(f"🚀 可以运行完整测试: python end_to_end_test.py")
        print(f"📚 可以运行使用示例: python geoclip_usage_example.py")

    else:
        print(f"\n❌ 测试失败，请检查错误信息并修复问题")
        print(f"💡 建议:")
        print(f"   1. 运行 python env_check.py 检查环境")
        print(f"   2. 检查是否所有依赖都已安装")
        print(f"   3. 检查import_test.py的结果")

    return success


if __name__ == "__main__":
    success = main()