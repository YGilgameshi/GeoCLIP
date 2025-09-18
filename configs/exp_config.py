import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Dict, Any, List, Optional, Tuple
import cv2
from pathlib import Path


# ================================
# 基础配置类
# ================================

class GeoCLIPConfig:
    """GeoCLIP配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._get_default_config()

        if config_path:
            self.load_config(config_path)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 模型配置
            'model': {
                'name': 'GeoCLIP',
                'backbone': 'ViT-L-14-336',
                'image_size': 336,
                'device': 'cuda'
            },

            # 深度估计配置
            'depth_estimator': {
                'type': 'single',  # single, multiscale, uncertainty
                'model_type': 'dpt_hybrid_384',
                'input_size': [384, 384],
                'device': 'cuda'
            },

            # 体素转换配置
            'voxel_converter': {
                'type': 'dense',  # dense, sparse, adaptive
                'voxel_size': 64,
                'world_size': 2.0,
                'min_depth': 0.1,
                'max_depth': 10.0,
                'device': 'cuda'
            },

            # 几何编码器配置
            'geometry_encoder': {
                'type': 'voxel',  # voxel, sparse, hierarchical, geometry_aware
                'in_channels': 3,
                'base_channels': 64,
                'num_stages': 4,
                'output_channels': 512,
                'voxel_size': 64
            },

            # 训练配置
            'training': {
                'batch_size': 4,
                'learning_rate': 1e-4,
                'num_epochs': 50,
                'weight_decay': 1e-5,
                'scheduler': 'cosine',
                'warmup_epochs': 5
            },

            # 数据配置
            'data': {
                'train_datasets': ['mvtec_3d', 'real3d_ad'],
                'test_datasets': ['mvtec_3d'],
                'data_root': './data',
                'num_workers': 4,
                'pin_memory': True
            },

            # 实验配置
            'experiment': {
                'name': 'geoclip_base',
                'output_dir': './outputs',
                'save_freq': 5,
                'eval_freq': 1,
                'log_freq': 100
            }
        }

    def load_config(self, config_path: str):
        """从文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)

        # 递归更新配置
        self._update_config(self.config, loaded_config)

    def _update_config(self, base_config: Dict, new_config: Dict):
        """递归更新配置"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value

    def save_config(self, save_path: str):
        """保存配置到文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def get(self, key_path: str, default=None):
        """获取配置值，支持点分割的路径"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """设置配置值"""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value