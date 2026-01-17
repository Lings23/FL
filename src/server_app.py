"""
Flower服务器应用
创建和配置服务器实例及聚合策略
"""
from typing import Dict
import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.serverapp_components import ServerAppComponents
from flwr.server.strategy import Strategy

from src.models import get_model, get_weights, set_weights, MODELS
from src.strategies import FedAvgStrategy, FedMedianStrategy, FedTrimmedMeanStrategy
from src.task import test, load_datasets, save_model, save_results
from src.config import get_config


def get_evaluate_fn(model, test_loader):
    """
    创建服务器端评估函数
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        
    Returns:
        evaluate_fn: 评估函数
    """
    def evaluate(server_round: int, parameters_ndarrays, config):
        """
        在服务器端评估全局模型
        
        Args:
            server_round: 当前轮次
            parameters_ndarrays: 模型参数
            config: 配置字典
            
        Returns:
            (loss, metrics): 损失和指标字典
        """
        # 设置模型参数
        set_weights(model, parameters_ndarrays)
        
        # 在测试集上评估
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss, accuracy = test(model, test_loader, device)
        
        print(f"\n[服务器评估 - 轮次 {server_round}]")
        print(f"  损失: {loss:.4f}")
        print(f"  准确率: {accuracy:.2f}%")
        
        return loss, {"centralized_accuracy": accuracy}
    
    return evaluate


def get_fit_config_fn():
    """
    创建fit配置函数
    
    Returns:
        fit_config_fn: 配置函数
    """
    config = get_config()
    
    def fit_config(server_round: int) -> Dict[str, float]:
        """
        配置客户端训练参数
        
        Args:
            server_round: 当前轮次
            
        Returns:
            config_dict: 配置字典
        """
        return {
            "lr": config.model['learning_rate'],
            "local_epochs": config.client['local_epochs']
        }
    
    return fit_config


def weighted_average(metrics):
    """
    计算加权平均指标
    
    Args:
        metrics: 客户端指标列表
        
    Returns:
        aggregated_metrics: 聚合后的指标字典
    """
    # 计算加权准确率
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def get_strategy() -> Strategy:
    """
    根据配置创建聚合策略
    
    Returns:
        strategy: Flower聚合策略
    """
    config = get_config()
    model_name = config.model['name']
    strategy_name = config.server.get('strategy', 'FedAvg')
    
    # 获取模型
    model = get_model(model_name)
    
    # 加载测试数据集
    _, test_dataset = load_datasets(model_name)
    batch_size = config.server.get('batch_size', 64)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # 策略参数
    strategy_kwargs = {
        'fraction_fit': config.server['fraction_fit'],
        'fraction_evaluate': config.server['fraction_eval'],
        'min_fit_clients': max(2, int(config.client['num_clients'] * config.server['fraction_fit'])),
        'min_evaluate_clients': max(2, int(config.client['num_clients'] * config.server['fraction_eval'])),
        'min_available_clients': config.client['num_clients'],
        'initial_parameters': ndarrays_to_parameters(get_weights(model)),
        'on_fit_config_fn': get_fit_config_fn(),
        'evaluate_fn': get_evaluate_fn(model, test_loader),
        'evaluate_metrics_aggregation_fn': weighted_average,
        'model': model,
        'save_path': 'outputs'
    }
    
    # 根据策略名称创建策略
    if strategy_name == 'FedAvg':
        strategy = FedAvgStrategy(**strategy_kwargs)
    elif strategy_name == 'FedMedian':
        strategy = FedMedianStrategy(**strategy_kwargs)
    elif strategy_name == 'FedTrimmedMean':
        beta = config.server.get('beta', 0.1)
        strategy = FedTrimmedMeanStrategy(beta=beta, **strategy_kwargs)
    else:
        raise ValueError(f"未知的策略: {strategy_name}")
    
    print(f"\n使用策略: {strategy_name}")
    return strategy


def get_server_fn():
    """
    创建服务器函数
    
    Returns:
        ServerApp: Flower服务器应用
    """
    def server_fn(context: Context):
        """
        服务器初始化函数
        
        Args:
            context: Flower上下文对象
            
        Returns:
            配置好的服务器组件
        """
        config = get_config()
        
        # 创建策略
        strategy = get_strategy()
        
        # 创建服务器配置
        server_config = ServerConfig(num_rounds=config.server['num_rounds'])
        
        # 返回 ServerAppComponents 对象
        return ServerAppComponents(
            config=server_config,
            strategy=strategy
        )
    
    return ServerApp(server_fn=server_fn)
