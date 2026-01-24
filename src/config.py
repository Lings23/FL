"""
配置管理模块
使用Pydantic进行配置验证和管理
"""
import os
from pathlib import Path
import yaml
from typing import Optional


class Config:
    """配置类，用于加载和管理YAML配置文件"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        
    def _load_config(self):
        """加载YAML配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @property
    def server(self):
        """服务器配置"""
        return self._config.get('server', {})
    
    @property
    def client(self):
        """客户端配置"""
        return self._config.get('client', {})
    
    @property
    def model(self):
        """模型配置"""
        return self._config.get('model', {})
    
    @property
    def model_configs(self):
        """模型特定配置"""
        return self._config.get('model_configs', {})
    
    def get_model_config(self, model_name: Optional[str] = None):
        """
        获取模型配置，支持模型特定配置覆盖
        
        Args:
            model_name: 模型名称，如果为None则使用self.model['name']
        
        Returns:
            合并后的模型配置字典
        """
        if model_name is None:
            model_name = self.model.get('name', 'MNIST')
        
        # 基础配置
        base_config = self.model.copy()
        
        # 获取模型特定配置
        model_specific = self.model_configs.get(model_name, {})
        
        # 合并配置：模型特定配置覆盖基础配置中的None值
        merged_config = {}
        for key, value in base_config.items():
            if value is None and key in model_specific:
                merged_config[key] = model_specific[key]
            else:
                merged_config[key] = value
        
        # 添加模型特定配置中的其他字段
        for key, value in model_specific.items():
            if key not in merged_config:
                merged_config[key] = value
        
        return merged_config
    
    @property
    def general(self):
        """通用配置"""
        return self._config.get('general', {})
    
    @property
    def backend(self):
        """后端配置"""
        return self._config.get('backend', {})
    
    @property
    def data(self):
        """数据配置"""
        return self._config.get('data', {})
    
    @property
    def attack(self):
        """攻击配置"""
        return self._config.get('attack', {})
    
    def __getitem__(self, key):
        """支持字典式访问"""
        return self._config[key]
    
    def get(self, key, default=None):
        """支持get方法"""
        return self._config.get(key, default)


# 全局配置实例
config = None


def init_config(config_path: str = "configs/config.yaml"):
    """初始化全局配置"""
    global config
    config = Config(config_path)
    return config


def get_config():
    """获取全局配置实例"""
    global config
    if config is None:
        config = init_config()
    return config
