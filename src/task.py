"""
训练和评估任务模块
包含模型训练、测试和数据加载功能
"""
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import datasets, transforms

from src.config import get_config
from src.models import MODELS


def train(model: nn.Module, train_loader: DataLoader, epochs: int, lr: float, device: str = 'cpu'):
    """
    训练模型
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 设备 ('cpu' 或 'cuda')
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Tuple[float, float]:
    """
    测试模型
    
    Args:
        model: 待测试的模型
        test_loader: 测试数据加载器
        device: 设备
        
    Returns:
        (loss, accuracy): 损失和准确率
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = loss / len(test_loader)
    
    return avg_loss, accuracy


def load_datasets(dataset_name: str, data_path: str = './datasets'):
    """
    加载数据集
    
    Args:
        dataset_name: 数据集名称 ('MNIST', 'FMNIST', 'CIFAR10')
        data_path: 数据集保存路径
        
    Returns:
        (train_dataset, test_dataset): 训练集和测试集
    """
    transform = MODELS[dataset_name]['transforms']
    
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'FMNIST':
        train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return train_dataset, test_dataset


def partition_data(dataset, num_clients: int, partition_type: str = 'iid', alpha: float = 0.5):
    """
    将数据集分区给多个客户端
    
    Args:
        dataset: PyTorch数据集
        num_clients: 客户端数量
        partition_type: 分区类型 ('iid' 或 'non_iid')
        alpha: Dirichlet分布参数 (仅用于non_iid)
        
    Returns:
        client_datasets: 客户端数据集列表
    """
    # 获取所有数据和标签
    if hasattr(dataset, 'data'):
        data = dataset.data.numpy() if isinstance(dataset.data, torch.Tensor) else dataset.data
        targets = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else np.array(dataset.targets)
    else:
        data = np.array([dataset[i][0].numpy() for i in range(len(dataset))])
        targets = np.array([dataset[i][1] for i in range(len(dataset))])
    
    num_samples = len(data)
    num_classes = len(np.unique(targets))
    
    if partition_type == 'iid':
        # IID分区：随机均匀分配
        indices = np.random.permutation(num_samples)
        client_indices = np.array_split(indices, num_clients)
    else:
        # Non-IID分区：使用Dirichlet分布
        client_indices = [[] for _ in range(num_clients)]
        
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            
            # 使用Dirichlet分布分配每个类别的样本
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < num_samples / num_clients) 
                                   for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            client_idx_k = np.split(idx_k, proportions)
            client_indices = [idx_j + idx.tolist() for idx_j, idx in zip(client_indices, client_idx_k)]
    
    # 创建客户端数据集
    client_datasets = []
    for indices in client_indices:
        if len(indices) > 0:
            client_data = data[indices]
            client_targets = targets[indices]
            
            # 转换为Tensor
            if isinstance(dataset.data, torch.Tensor):
                client_data = torch.from_numpy(client_data)
                client_targets = torch.from_numpy(client_targets)
            
            client_datasets.append((client_data, client_targets))
    
    return client_datasets


def create_data_loaders(client_data, client_targets, batch_size: int, train_split: float = 0.8):
    """
    为客户端创建训练和验证数据加载器
    
    Args:
        client_data: 客户端数据
        client_targets: 客户端标签
        batch_size: 批量大小
        train_split: 训练集比例
        
    Returns:
        (train_loader, val_loader): 训练和验证数据加载器
    """
    # 分割训练集和验证集
    num_train = int(len(client_data) * train_split)
    
    # 转换为Tensor
    if not isinstance(client_data, torch.Tensor):
        client_data = torch.FloatTensor(client_data)
    if not isinstance(client_targets, torch.Tensor):
        client_targets = torch.LongTensor(client_targets)
    
    # 确保数据维度正确 (添加通道维度如果需要)
    if len(client_data.shape) == 3:  # (N, H, W)
        client_data = client_data.unsqueeze(1)  # (N, 1, H, W)
    
    # 归一化
    client_data = client_data.float() / 255.0
    
    train_data = client_data[:num_train]
    train_targets = client_targets[:num_train]
    val_data = client_data[num_train:]
    val_targets = client_targets[num_train:]
    
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def save_results(results: dict, save_dir: str = 'outputs'):
    """
    保存实验结果
    
    Args:
        results: 结果字典
        save_dir: 保存目录
    """
    # 创建时间戳目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(save_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存结果为JSON
    with open(output_path / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return str(output_path)


def save_model(model: nn.Module, save_path: str, round_num: int, accuracy: float):
    """
    保存模型检查点
    
    Args:
        model: 模型
        save_path: 保存路径
        round_num: 轮次
        accuracy: 准确率
    """
    filename = f"model_round_{round_num}_acc_{accuracy:.2f}.pth"
    filepath = Path(save_path) / filename
    torch.save(model.state_dict(), filepath)
    print(f"模型已保存: {filepath}")
