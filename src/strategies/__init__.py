"""
策略模块初始化
"""
from src.strategies.fed_avg import FedAvgStrategy, FedMedianStrategy, FedTrimmedMeanStrategy

__all__ = [
    'FedAvgStrategy',
    'FedMedianStrategy',
    'FedTrimmedMeanStrategy'
]
