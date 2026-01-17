"""
初始化文件
"""
from src.models import get_model, get_weights, set_weights, MODELS
from src.config import get_config, init_config
from src.task import train, test, load_datasets

__all__ = [
    'get_model',
    'get_weights', 
    'set_weights',
    'MODELS',
    'get_config',
    'init_config',
    'train',
    'test',
    'load_datasets'
]
