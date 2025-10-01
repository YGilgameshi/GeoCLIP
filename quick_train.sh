#!/bin/bash

# GeoCLIP 快速训练脚本
# 基于AdaCLIP的数据集组合进行GeoCLIP训练

echo "🚀 GeoCLIP快速训练脚本"
echo "======================"

# 检查GPU可用性
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ GPU可用"
    DEVICE="cuda"
else
    echo "⚠️ GPU不可用，将使用CPU（速度较慢）"
    DEVICE="cpu"
fi

# 检查Python环境
if python -c "import torch; import geoclip" > /dev/null 2>&1; then
    echo "✅ GeoCLIP环境正常"
else
    echo "❌ GeoCLIP环境异常，请先运行 python import_test.py 检查"
    exit 1
fi

# 设置默认参数
SCENARIO=${1:-"standard"}  # 训练场景：debug, standard, industrial, medical
GPU_ID=${2:-"0"}           # GPU ID

echo "📋 训练场景: $SCENARIO"
echo "🎮 GPU ID: $GPU_ID"

# 导出CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 根据场景选择参数
case $SCENARIO in
    "debug")
        echo "🐛 调试模式 - 快速验证"
        python train_geoclip.py \
            --training_data mvtec \
            --testing_data visa \
            --model ViT-B-16 \
            --epoch 5 \
            --batch_size 4 \
            --learning_rate 1e-3 \
            --fusion_type simple_concat \
            --geometry_encoder voxel \
            --print_freq 2 \
            --save_freq 2 \
            --preprocess_depth False \
            --seed 42
        ;;

    "standard")
        echo "📊 标准模式 - AdaCLIP原始组合"
        python train_geoclip.py \
            --training_data mvtec colondb \
            --testing_data visa \
            --model ViT-L-14 \
            --epoch 100 \
            --batch_size 8 \
            --learning_rate 1e-4 \
            --fusion_type cross_attention \
            --geometry_encoder voxel \
            --fusion_dim 1024 \
            --contrastive_weight 1.0 \
            --geometry_weight 0.5 \
            --anomaly_weight 2.0 \
            --preprocess_depth True \
            --print_freq 10 \
            --save_freq 5 \
            --early_stopping_patience 15 \
            --seed 111
        ;;

    "industrial")
        echo "🏭 工业模式 - 专注工业异常检测"
        python train_geoclip.py \
            --training_data mvtec visa \
            --testing_data btad \
            --model ViT-L-14-336 \
            --epoch 150 \
            --batch_size 8 \
            --learning_rate 5e-5 \
            --fusion_type cross_attention \
            --geometry_encoder hierarchical \
            --fusion_dim 1024 \
            --contrastive_weight 1.5 \
            --geometry_weight 0.8 \
            --anomaly_weight 2.5 \
            --preprocess_depth True \
            --freeze_clip False \
            --print_freq 5 \
            --save_freq 10 \
            --early_stopping_patience 20 \
            --seed 111
        ;;

    "medical")
        echo "🏥 医学模式 - 专注医学异常检测"
        python train_geoclip.py \
            --training_data colondb clinicdb \
            --testing_data isic \
            --model ViT-L-14 \
            --epoch 120 \
            --batch_size 6 \
            --learning_rate 8e-5 \
            --fusion_type adaptive_fusion \
            --geometry_encoder voxel \
            --fusion_dim 512 \
            --contrastive_weight 0.8 \
            --geometry_weight 1.2 \
            --anomaly_weight 2.0 \
            --preprocess_depth True \
            --print_freq 8 \
            --save_freq 5 \
            --early_stopping_patience 18 \
            --seed 111
        ;;

    "mixed")
        echo "🎯 混合模式 - 多数据集训练"
        python train_geoclip.py \
            --training_data mvtec colondb visa \
            --testing_data btad clinicdb \
            --model ViT-L-14-336 \
            --epoch 200 \
            --batch_size 12 \
            --learning_rate 3e-5 \
            --fusion_type cross_attention \
            --geometry_encoder hierarchical \
            --fusion_dim 1024 \
            --contrastive_weight 1.2 \
            --geometry_weight 0.6 \
            --anomaly_weight 2.2 \
            --preprocess_depth True \
            --use_wandb True \
            --print_freq 5 \
            --save_freq 10 \
            --early_stopping_patience 25 \
            --seed 111
        ;;

    "test")
        echo "🧪 测试模式 - 单类别快速测试"
        python train_geoclip.py \
            --training_data mvtec \
            --testing_data mvtec \
            --model ViT-B-16 \
            --epoch 3 \
            --batch_size 2 \
            --learning_rate 1e-3 \
            --fusion_type simple_concat \
            --geometry_encoder voxel \
            --print_freq 1 \
            --save_freq 1 \
            --preprocess_depth False \
            --seed 42
        ;;

    *)
        echo "❌ 未知场景: $SCENARIO"
        echo ""
        echo "可用场景:"
        echo "  debug     - 调试模式，快速验证"
        echo "  standard  - 标准模式，AdaCLIP原始组合"
        echo "  industrial- 工业模式，专注工业数据"
        echo "  medical   - 医学模式，专注医学数据"
        echo "  mixed     - 混合模式，多数据集训练"
        echo "  test      - 测试模式，最小配置"
        echo ""
        echo "用法: $0 <场景> [GPU_ID]"
        echo "示例: $0 standard 0"
        exit 1
        ;;
esac

echo ""
echo "✅ 训练脚本执行完成！"
echo "📁 结果保存在: ./workspaces_geoclip/"
echo "📊 可以使用TensorBoard查看训练过程:"
echo "   tensorboard --logdir ./workspaces_geoclip/"