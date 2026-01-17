"""
Flower客户端应用
创建和配置客户端实例
"""
from flwr.client import Client, ClientApp
from flwr.common import Context

from src.flower_client import FlowerClient
from src.models import MODELS, get_model
from src.task import partition_data, load_datasets, create_data_loaders
from src.config import get_config


def get_client_fn():
    """
    创建客户端函数
    
    Returns:
        ClientApp: Flower客户端应用
    """
    config = get_config()
    model_name = config.model['name']
    batch_size = config.client['batch_size']
    
    # 加载数据集
    train_dataset, test_dataset = load_datasets(model_name)
    
    # 数据分区
    num_clients = config.client['num_clients']
    partition_type = config.data.get('partitioning', 'iid')
    alpha = config.data.get('alpha', 0.5)
    
    client_datasets = partition_data(train_dataset, num_clients, partition_type, alpha)
    
    def client_fn(context: Context) -> Client:
        """
        客户端初始化函数
        
        Args:
            context: Flower上下文对象
            
        Returns:
            Client: 客户端实例
        """
        # 获取客户端ID和总数
        partition_id = context.node_config["partition-id"]
        
        # 获取模型
        model = get_model(model_name)
        
        # 获取客户端数据
        client_data, client_targets = client_datasets[partition_id]
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            client_data, 
            client_targets, 
            batch_size
        )
        
        # 创建客户端实例
        client_instance = FlowerClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            client_id=partition_id
        )
        
        return client_instance.to_client()
    
    return ClientApp(client_fn=client_fn)
