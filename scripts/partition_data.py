"""
数据分区脚本
将数据集分区并保存到本地，供后续使用
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torchvision import datasets

from src.task import load_datasets, partition_data


def save_partitions(dataset_name: str, num_clients: int = 10, 
                   partition_type: str = 'iid', alpha: float = 0.5):
    """
    将数据集分区并保存到本地
    
    Args:
        dataset_name: 数据集名称 ('MNIST', 'FMNIST', 'CIFAR10')
        num_clients: 客户端数量
        partition_type: 分区类型 ('iid' 或 'non_iid')
        alpha: Dirichlet分布参数
    """
    print(f"正在分区数据集: {dataset_name}")
    print(f"  客户端数量: {num_clients}")
    print(f"  分区类型: {partition_type}")
    if partition_type == 'non_iid':
        print(f"  Alpha参数: {alpha}")
    
    # 加载数据集
    train_dataset, _ = load_datasets(dataset_name)
    
    # 分区
    client_datasets = partition_data(train_dataset, num_clients, partition_type, alpha)
    
    # 创建保存目录
    save_dir = Path('data') / 'client' / dataset_name / f'num_clients_{num_clients}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每个客户端的数据
    for i, (client_data, client_targets) in enumerate(client_datasets):
        client_file = save_dir / f'client_{i}.npz'
        
        # 转换为numpy数组
        if isinstance(client_data, torch.Tensor):
            client_data = client_data.numpy()
        if isinstance(client_targets, torch.Tensor):
            client_targets = client_targets.numpy()
        
        np.savez_compressed(client_file, data=client_data, targets=client_targets)
        print(f"  客户端 {i}: {len(client_data)} 个样本 -> {client_file}")
    
    # 保存分区信息
    info = {
        'dataset': dataset_name,
        'num_clients': num_clients,
        'partition_type': partition_type,
        'alpha': alpha if partition_type == 'non_iid' else None,
        'samples_per_client': [len(data) for data, _ in client_datasets]
    }
    
    import json
    info_file = save_dir / 'partition_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
    
    print(f"\n分区完成! 数据已保存到: {save_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据集分区工具")
    parser.add_argument(
        "dataset",
        type=str,
        choices=['MNIST', 'FMNIST', 'CIFAR10'],
        help="数据集名称"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=10,
        help="客户端数量 (默认: 10)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=['iid', 'non_iid'],
        default='iid',
        help="分区类型 (默认: iid)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet分布参数 (仅用于non_iid, 默认: 0.5)"
    )
    
    args = parser.parse_args()
    
    save_partitions(
        dataset_name=args.dataset,
        num_clients=args.num_clients,
        partition_type=args.type,
        alpha=args.alpha
    )


if __name__ == "__main__":
    main()
