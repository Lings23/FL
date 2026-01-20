"""
联邦学习攻击模块
包含各种针对联邦学习的攻击方法
"""
import torch
import torch.nn as nn
from typing import Optional


def flip_labels(labels: torch.Tensor, total_number_classes: int) -> torch.Tensor:
    """
    标签翻转攻击
    Flips the given labels across the range of classes.
    For a label `y` in a classification task with `C` classes, the flipped label becomes `C - 1 - y`.
    
    Args:
        labels: 需要被翻转的类别标签张量
        total_number_classes: 分类任务中的总类别数
    
    Returns:
        翻转后的标签张量
    """
    flipped_tensor = total_number_classes - 1 - labels
    return flipped_tensor


def add_gaussian_noise(parameters, mean: float = 0.0, std: float = 1.0):
    """
    高斯噪声攻击
    Adds Gaussian noise with user-defined mean and standard deviation to model parameters.
    
    Args:
        parameters: 模型参数
        mean: 高斯噪声均值
        std: 高斯噪声标准差
    """
    for param in parameters:
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * std + mean
            param.grad += noise


def flip_sign(parameters):
    """
    符号翻转攻击（梯度反转攻击）
    Flips sign of gradient for model parameters.
    
    Args:
        parameters: 模型参数
    """
    for param in parameters:
        if param.grad is not None:
            param.grad *= -1  # 翻转梯度方向


def scale_attack(parameters, scale_factor: float = 10.0):
    """
    梯度缩放攻击
    Scales the gradients by a factor to amplify the malicious update.
    
    Args:
        parameters: 模型参数
        scale_factor: 缩放因子
    """
    for param in parameters:
        if param.grad is not None:
            param.grad *= scale_factor


def zero_gradient_attack(parameters):
    """
    零梯度攻击
    Sets all gradients to zero, effectively making the client not contribute.
    
    Args:
        parameters: 模型参数
    """
    for param in parameters:
        if param.grad is not None:
            param.grad.zero_()


def random_update_attack(parameters):
    """
    随机更新攻击
    Replaces gradients with random values.
    
    Args:
        parameters: 模型参数
    """
    for param in parameters:
        if param.grad is not None:
            param.grad = torch.randn_like(param.grad)


# 攻击方法注册表
ATTACK_REGISTRY = {
    'flip_labels': flip_labels,
    'gaussian_noise': add_gaussian_noise,
    'flip_sign': flip_sign,
    'scale': scale_attack,
    'zero_gradient': zero_gradient_attack,
    'random_update': random_update_attack,
}


def get_attack_function(attack_name: str):
    """
    根据攻击名称获取对应的攻击函数
    
    Args:
        attack_name: 攻击方法名称
    
    Returns:
        对应的攻击函数，如果不存在则返回None
    """
    return ATTACK_REGISTRY.get(attack_name, None)
