"""
GeoCLIP Configuration File
管理数据集路径和全局配置
"""

import os
from pathlib import Path

# ================================
# 全局数据集配置
# ================================

# 数据集根目录 - 请根据您的实际情况修改
DATA_ROOT = './geoclip/datasets'  # 默认AdaCLIP数据集路径

# 如果环境变量中定义了数据集根目录，使用环境变量
if 'GEOCLIP_DATA_ROOT' in os.environ:
    DATA_ROOT = os.environ['GEOCLIP_DATA_ROOT']

# 确保数据集路径存在
DATA_ROOT = Path(DATA_ROOT).absolute()
if not DATA_ROOT.exists():
    print(f"警告: 数据集根目录不存在: {DATA_ROOT}")
    print("请设置正确的DATA_ROOT路径或设置环境变量GEOCLIP_DATA_ROOT")

# ================================
# 各个数据集的具体路径
# ================================

# 工业异常检测数据集
MVTEC_ROOT = DATA_ROOT / 'MVTec_AD'
VISA_ROOT = DATA_ROOT / 'VisA'
BTAD_ROOT = DATA_ROOT / 'BTAD'
MPDD_ROOT = DATA_ROOT / 'MPDD'
SDD_ROOT = DATA_ROOT / 'SDD_anomaly_detection'

# 医学异常检测数据集
COLONDB_ROOT = DATA_ROOT / 'CVC-ColonDB'
CLINICDB_ROOT = DATA_ROOT / 'CVC-ClinicDB'
ISIC_ROOT = DATA_ROOT / 'ISIC2018'
BRAIN_MRI_ROOT = DATA_ROOT / 'brain_mri'
HEAD_CT_ROOT = DATA_ROOT / 'head_ct'

# 其他数据集
DTD_ROOT = DATA_ROOT / 'dtd'
DAGM_ROOT = DATA_ROOT / 'DAGM'
BR35H_ROOT = DATA_ROOT / 'br35h'
TN3K_ROOT = DATA_ROOT / 'TN3K'

# ================================
# 数据集类别定义
# ================================

# MVTec AD 类别
MVTEC_CLS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# VisA 类别
VISA_CLS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum'
]

# BTAD 类别
BTAD_CLS_NAMES = ['01', '02', '03']

# 医学数据集类别
COLONDB_CLS_NAMES = ['ColonDB']
CLINICDB_CLS_NAMES = ['ClinicDB']

# ================================
# GeoCLIP特定配置
# ================================

# 深度缓存配置
DEPTH_CACHE_ROOT = Path('./depth_cache')
DEPTH_CACHE_ROOT.mkdir(exist_ok=True)

# 默认深度估计器配置
DEFAULT_DEPTH_CONFIG = {
    'model_type': 'DPT_Large',
    'device': 'cuda',
    'input_size': (384, 384)
}

# 默认体素转换配置
DEFAULT_VOXEL_CONFIG = {
    'voxel_size': 64,
    'depth_range': (0.1, 10.0),
    'spatial_range': (-2.0, 2.0),
    'use_color': True
}

# 默认几何编码器配置
DEFAULT_GEOMETRY_CONFIG = {
    'type': 'voxel',
    'in_channels': 4,  # RGB + Depth
    'base_channels': 64,
    'num_stages': 4,
    'output_channels': 512,
    'voxel_size': 64
}

# ================================
# 训练配置预设
# ================================

# 快速训练配置（用于调试）
QUICK_TRAIN_CONFIG = {
    'epoch': 10,
    'batch_size': 4,
    'learning_rate': 1e-3,
    'print_freq': 2,
    'save_freq': 5,
    'preprocess_depth': False
}

# 标准训练配置
STANDARD_TRAIN_CONFIG = {
    'epoch': 100,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'print_freq': 10,
    'save_freq': 5,
    'preprocess_depth': True
}

# 高质量训练配置
HIGH_QUALITY_TRAIN_CONFIG = {
    'epoch': 200,
    'batch_size': 16,
    'learning_rate': 5e-5,
    'print_freq': 5,
    'save_freq': 10,
    'preprocess_depth': True
}

# ================================
# 数据集组合预设
# ================================

# AdaCLIP原始组合
ADACLIP_TRAIN_COMBO_1 = {
    'training_data': ['mvtec', 'colondb'],
    'testing_data': ['visa']
}

ADACLIP_TRAIN_COMBO_2 = {
    'training_data': ['visa', 'clinicdb'],
    'testing_data': ['mvtec']
}

# GeoCLIP推荐组合
GEOCLIP_INDUSTRIAL_COMBO = {
    'training_data': ['mvtec', 'visa'],
    'testing_data': ['btad']
}

GEOCLIP_MEDICAL_COMBO = {
    'training_data': ['colondb', 'clinicdb'],
    'testing_data': ['isic']
}

GEOCLIP_MIXED_COMBO = {
    'training_data': ['mvtec', 'colondb', 'visa'],
    'testing_data': ['btad', 'clinicdb']
}

# ================================
# 模型配置预设
# ================================

# 轻量级配置
LIGHT_MODEL_CONFIG = {
    'model': 'ViT-B-16',
    'geometry_encoder': 'voxel',
    'fusion_type': 'simple_concat',
    'fusion_dim': 512,
    'freeze_clip': False
}

# 标准配置
STANDARD_MODEL_CONFIG = {
    'model': 'ViT-L-14',
    'geometry_encoder': 'voxel',
    'fusion_type': 'cross_attention',
    'fusion_dim': 1024,
    'freeze_clip': False
}

# 高端配置
PREMIUM_MODEL_CONFIG = {
    'model': 'ViT-L-14-336',
    'geometry_encoder': 'hierarchical',
    'fusion_type': 'cross_attention',
    'fusion_dim': 1024,
    'freeze_clip': False
}


# ================================
# 实用函数
# ================================

def get_dataset_info(dataset_name: str) -> dict:
    """获取数据集信息"""
    dataset_info_map = {
        'mvtec': {
            'root': MVTEC_ROOT,
            'classes': MVTEC_CLS_NAMES,
            'type': 'industrial',
            'description': 'MVTec AD - Industrial Anomaly Detection'
        },
        'visa': {
            'root': VISA_ROOT,
            'classes': VISA_CLS_NAMES,
            'type': 'industrial',
            'description': 'VisA - Visual Anomaly Detection'
        },
        'btad': {
            'root': BTAD_ROOT,
            'classes': BTAD_CLS_NAMES,
            'type': 'industrial',
            'description': 'BTAD - Beantech Anomaly Detection'
        },
        'colondb': {
            'root': COLONDB_ROOT,
            'classes': COLONDB_CLS_NAMES,
            'type': 'medical',
            'description': 'CVC-ColonDB - Colon Polyp Detection'
        },
        'clinicdb': {
            'root': CLINICDB_ROOT,
            'classes': CLINICDB_CLS_NAMES,
            'type': 'medical',
            'description': 'CVC-ClinicDB - Clinical Polyp Detection'
        }
    }

    return dataset_info_map.get(dataset_name, {})


def check_dataset_availability():
    """检查数据集可用性"""
    print("🔍 检查数据集可用性...")
    print(f"数据集根目录: {DATA_ROOT}")

    datasets = {
        'mvtec': MVTEC_ROOT,
        'visa': VISA_ROOT,
        'btad': BTAD_ROOT,
        'colondb': COLONDB_ROOT,
        'clinicdb': CLINICDB_ROOT,
    }

    available = []
    missing = []

    for name, path in datasets.items():
        if path.exists():
            available.append(name)
            print(f"✅ {name}: {path}")
        else:
            missing.append(name)
            print(f"❌ {name}: {path} (不存在)")

    print(f"\n📊 统计:")
    print(f"✅ 可用数据集: {len(available)} ({', '.join(available)})")
    print(f"❌ 缺失数据集: {len(missing)} ({', '.join(missing)})")

    if missing:
        print(f"\n💡 下载缺失的数据集:")
        for name in missing:
            info = get_dataset_info(name)
            if info:
                print(f"   {name}: {info['description']}")

    return available, missing


def get_recommended_config(scenario: str = 'standard') -> dict:
    """获取推荐的训练配置"""
    if scenario == 'debug':
        return {
            **QUICK_TRAIN_CONFIG,
            **LIGHT_MODEL_CONFIG,
            **ADACLIP_TRAIN_COMBO_1
        }
    elif scenario == 'standard':
        return {
            **STANDARD_TRAIN_CONFIG,
            **STANDARD_MODEL_CONFIG,
            **ADACLIP_TRAIN_COMBO_1
        }
    elif scenario == 'high_quality':
        return {
            **HIGH_QUALITY_TRAIN_CONFIG,
            **PREMIUM_MODEL_CONFIG,
            **GEOCLIP_MIXED_COMBO
        }
    elif scenario == 'industrial':
        return {
            **STANDARD_TRAIN_CONFIG,
            **STANDARD_MODEL_CONFIG,
            **GEOCLIP_INDUSTRIAL_COMBO
        }
    elif scenario == 'medical':
        return {
            **STANDARD_TRAIN_CONFIG,
            **STANDARD_MODEL_CONFIG,
            **GEOCLIP_MEDICAL_COMBO
        }
    else:
        raise ValueError(f"未知场景: {scenario}")


def setup_environment():
    """设置环境变量和路径"""
    # 设置CUDA相关环境变量
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    # 添加当前项目到Python路径
    current_dir = Path(__file__).parent.absolute()
    if str(current_dir) not in os.sys.path:
        os.sys.path.insert(0, str(current_dir))

    # 创建必要的目录
    DEPTH_CACHE_ROOT.mkdir(exist_ok=True)

    print("🔧 环境设置完成")
    print(f"   项目路径: {current_dir}")
    print(f"   数据根目录: {DATA_ROOT}")
    print(f"   深度缓存目录: {DEPTH_CACHE_ROOT}")


def print_config_summary():
    """打印配置摘要"""
    print("\n📋 GeoCLIP配置摘要")
    print("=" * 40)
    print(f"数据根目录: {DATA_ROOT}")
    print(f"深度缓存目录: {DEPTH_CACHE_ROOT}")

    print(f"\n支持的数据集:")
    for name in ['mvtec', 'visa', 'btad', 'colondb', 'clinicdb']:
        info = get_dataset_info(name)
        status = "✅" if info['root'].exists() else "❌"
        print(f"  {status} {name}: {info['description']}")

    print(f"\n预设配置:")
    scenarios = ['debug', 'standard', 'high_quality', 'industrial', 'medical']
    for scenario in scenarios:
        try:
            config = get_recommended_config(scenario)
            print(f"  📋 {scenario}: {config.get('model', 'N/A')} + {'/'.join(config.get('training_data', []))}")
        except:
            print(f"  ❌ {scenario}: 配置错误")


# ================================
# 快速启动函数
# ================================

def quick_start_debug():
    """快速开始调试模式训练"""
    config = get_recommended_config('debug')
    print("🚀 快速启动 - 调试模式")
    print(f"配置: {config}")
    return config


def quick_start_standard():
    """快速开始标准训练"""
    config = get_recommended_config('standard')
    print("🚀 快速启动 - 标准模式")
    print(f"配置: {config}")
    return config


# ================================
# 自动检测和建议
# ================================

def auto_detect_best_config():
    """自动检测最佳配置"""
    available_datasets, missing_datasets = check_dataset_availability()

    # 根据可用数据集推荐配置
    if len(available_datasets) >= 3:
        return get_recommended_config('high_quality')
    elif 'mvtec' in available_datasets and 'colondb' in available_datasets:
        return get_recommended_config('standard')
    elif len(available_datasets) >= 1:
        return get_recommended_config('debug')
    else:
        print("❌ 没有可用的数据集，请先下载数据集")
        return None


# 主函数：如果直接运行此文件，执行配置检查
if __name__ == "__main__":
    print("🔧 GeoCLIP配置检查")
    print("=" * 50)

    # 设置环境
    setup_environment()

    # 检查数据集
    check_dataset_availability()

    # 打印配置摘要
    print_config_summary()

    # 推荐最佳配置
    print("\n💡 推荐配置:")
    best_config = auto_detect_best_config()
    if best_config:
        print("根据您的数据集情况，推荐使用以下配置开始训练:")
        print(f"python train_geoclip.py \\")
        print(f"  --training_data {' '.join(best_config['training_data'])} \\")
        print(f"  --testing_data {' '.join(best_config['testing_data'])} \\")
        print(f"  --model {best_config['model']} \\")
        print(f"  --epoch {best_config['epoch']} \\")
        print(f"  --batch_size {best_config['batch_size']} \\")
        print(f"  --learning_rate {best_config['learning_rate']}")

    print("\n✅ 配置检查完成！")