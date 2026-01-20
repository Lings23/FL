"""
攻击管理器
用于动态管理和应用各种攻击方法
"""
import random
from typing import List, Optional, Dict, Any
import torch

from src.attacks import get_attack_function


class AttackManager:
    """攻击管理器类，负责根据配置动态加载和执行攻击"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化攻击管理器
        
        Args:
            config: 攻击配置字典
        """
        self.enabled = config.get('enabled', False)
        self.attack_type = config.get('type', None)
        self.malicious_clients = config.get('malicious_clients', [])
        self.malicious_ratio = config.get('malicious_ratio', 0.0)
        self.params = config.get('params', {})
        
        # 动态确定恶意客户端列表
        if not self.malicious_clients and self.malicious_ratio > 0:
            self._initialize_malicious_clients()
    
    def _initialize_malicious_clients(self):
        """根据恶意比例初始化恶意客户端列表"""
        # 这个方法会在知道总客户端数量后被调用
        pass
    
    def set_malicious_clients_by_ratio(self, total_clients: int):
        """
        根据恶意客户端比例设置恶意客户端ID列表
        
        Args:
            total_clients: 总客户端数量
        """
        if self.malicious_ratio > 0 and not self.malicious_clients:
            num_malicious = int(total_clients * self.malicious_ratio)
            # Ensure the number of malicious clients does not exceed total_clients
            if num_malicious > total_clients:
                num_malicious = total_clients
            all_client_ids = list(range(total_clients))
            self.malicious_clients = random.sample(all_client_ids, num_malicious)
            print(f"[AttackManager] Malicious clients: {self.malicious_clients}")
    
    def is_malicious_client(self, client_id: int) -> bool:
        """
        判断客户端是否为恶意客户端
        
        Args:
            client_id: 客户端ID
        
        Returns:
            是否为恶意客户端
        """
        if not self.enabled:
            return False
        return client_id in self.malicious_clients
    
    def apply_attack(self, attack_target, **kwargs):
        """
        应用攻击
        
        Args:
            attack_target: 攻击目标（可以是模型参数、标签等）
            **kwargs: 其他参数
        
        Returns:
            攻击后的结果
        """
        if not self.enabled or not self.attack_type:
            return attack_target
        
        attack_func = get_attack_function(self.attack_type)
        if attack_func is None:
            print(f"[Warning] Attack type '{self.attack_type}' not found!")
            return attack_target
        
        # 根据攻击类型应用不同的攻击
        if self.attack_type == 'flip_labels':
            # 标签翻转攻击
            num_classes = kwargs.get('num_classes', self.params.get('num_classes', 10))
            return attack_func(attack_target, num_classes)
        
        elif self.attack_type == 'gaussian_noise':
            # 高斯噪声攻击
            mean = self.params.get('mean', 0.0)
            std = self.params.get('std', 1.0)
            attack_func(attack_target, mean, std)
            return None  # 直接修改参数
        
        elif self.attack_type == 'flip_sign':
            # 符号翻转攻击
            attack_func(attack_target)
            return None  # 直接修改参数
        
        elif self.attack_type == 'scale':
            # 缩放攻击
            scale_factor = self.params.get('scale_factor', 10.0)
            attack_func(attack_target, scale_factor)
            return None
        
        elif self.attack_type in ['zero_gradient', 'random_update']:
            # 零梯度或随机更新攻击
            attack_func(attack_target)
            return None
        
        return attack_target
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取攻击管理器信息
        
        Returns:
            攻击配置信息字典
        """
        return {
            'enabled': self.enabled,
            'attack_type': self.attack_type,
            'malicious_clients': self.malicious_clients,
            'malicious_ratio': self.malicious_ratio,
            'params': self.params
        }


def create_attack_manager(config: Dict[str, Any]) -> AttackManager:
    """
    工厂函数：创建攻击管理器实例
    
    Args:
        config: 攻击配置字典
    
    Returns:
        攻击管理器实例
    """
    return AttackManager(config)
