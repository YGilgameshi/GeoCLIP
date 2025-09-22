"""
GeoCLIP 数据集模块
"""

from .adaclip_adapter import (
    AdaCLIPToGeoCLIPAdapter,
    GeoCLIP_AdaCLIPDataset,
    create_geoclip_dataset
)

# 导入数据集工厂函数
try:
    from .dataset_factory import DatasetFactory, get_dataset
except ImportError:
    pass

__all__ = [
    'AdaCLIPToGeoCLIPAdapter',
    'GeoCLIP_AdaCLIPDataset',
    'create_geoclip_dataset',
    'DatasetFactory',
    'get_dataset'
]