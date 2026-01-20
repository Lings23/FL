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
from src.attack_manager import create_attack_manager


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
    
    # 创建攻击管理器（根据配置动态加载）
    attack_config = config.attack
    attack_manager = None
    if attack_config.get('enabled', False):
        attack_manager = create_attack_manager(attack_config)
        attack_manager.set_malicious_clients_by_ratio(num_clients)
        print(f"[AttackManager] Initialized with attack type: {attack_manager.attack_type}")
    
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
        
        # 创建客户端实例（传递攻击管理器）
        client_instance = FlowerClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            client_id=partition_id,
            attack_manager=attack_manager
        )
        
        return client_instance.to_client()
    
    return ClientApp(client_fn=client_fn)
