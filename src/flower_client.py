"""
Flower客户端实现
实现了基于Flower框架的联邦学习客户端
"""
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from typing import Dict, Tuple

from src.models import get_weights, set_weights
from src.task import train, test


class FlowerClient(NumPyClient):
    """
    Flower客户端类
    
    实现了fit和evaluate方法用于本地训练和评估
    """
    
    def __init__(self, model: torch.nn.Module, train_loader, val_loader, client_id: int):
        """
        初始化客户端
        
        Args:
            model: PyTorch模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            client_id: 客户端ID
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        本地训练
        
        Args:
            parameters: 全局模型参数
            config: 配置字典，包含学习率、训练轮数等
            
        Returns:
            (updated_parameters, num_examples, metrics): 更新后的参数、样本数和指标
        """
        # 设置模型参数为全局参数
        set_weights(self.model, parameters)
        
        # 从配置中获取训练参数
        lr = config.get("lr", 0.001)
        local_epochs = config.get("local_epochs", 1)
        
        # 本地训练
        train(self.model, self.train_loader, local_epochs, lr, self.device)
        
        # 返回更新后的参数
        return (
            get_weights(self.model),
            len(self.train_loader.dataset),
            {"client_id": self.client_id}
        )
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        本地评估
        
        Args:
            parameters: 全局模型参数
            config: 配置字典
            
        Returns:
            (loss, num_examples, metrics): 损失、样本数和指标
        """
        # 设置模型参数为全局参数
        set_weights(self.model, parameters)
        
        # 在验证集上评估
        loss, accuracy = test(self.model, self.val_loader, self.device)
        
        return (
            loss,
            len(self.val_loader.dataset),
            {"accuracy": accuracy, "client_id": self.client_id}
        )
