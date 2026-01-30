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


# 全局 History 记录器，用于收集训练结果
class TrainingHistory:
    """训练历史记录器"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.reset()
        return cls._instance
    
    def reset(self):
        """重置历史记录"""
        self.rounds = []
        self.losses = []
        self.accuracies = []
        self.config_info = {}
    
    def add_result(self, round_num: int, loss: float, accuracy: float):
        """添加一轮结果"""
        self.rounds.append(round_num)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def to_dict(self):
        """转换为字典"""
        return {
            "config": self.config_info,
            "rounds": self.rounds,
            "centralized_loss": self.losses,
            "centralized_accuracy": self.accuracies,
            "best_accuracy": max(self.accuracies) if self.accuracies else 0.0,
            "final_accuracy": self.accuracies[-1] if self.accuracies else 0.0,
            "final_loss": self.losses[-1] if self.losses else 0.0
        }


def get_training_history() -> TrainingHistory:
    """获取全局训练历史记录器"""
    return TrainingHistory()


def get_evaluate_fn(model, test_loader, num_rounds: int = 10):
    """
    创建服务器端评估函数
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        num_rounds: 总轮数，用于在最后一轮保存结果
        
    Returns:
        evaluate_fn: 评估函数
    """
    history = get_training_history()
    
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
        # 确保模型在正确设备上
        model.to(device)
        loss, accuracy = test(model, test_loader, device)
        
        print(f"\n[服务器评估 - 轮次 {server_round}]")
        print(f"  损失: {loss:.4f}")
        print(f"  准确率: {accuracy:.2f}%")
        print(f"  设备: {device}")
        
        # 记录到 history
        history.add_result(server_round, loss, accuracy)
        
        # 在最后一轮保存结果
        if server_round == num_rounds:
            _save_training_history(history)
        
        return loss, {"centralized_accuracy": accuracy}
    
    return evaluate


def _save_training_history(history: TrainingHistory):
    """保存训练历史到文件"""
    import json
    import datetime
    from pathlib import Path
    
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = history.config_info.get('model_name', 'results')
    json_path = out_dir / f"history_{model_name}_{ts}.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"\n训练历史已保存: {json_path}")


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
        # 获取模型特定配置
        model_config = config.get_model_config()
        
        return {
            "lr": float(model_config.get('learning_rate', 0.001)),
            "local_epochs": int(model_config.get('local_epochs', config.client.get('local_epochs', 1))),
            "weight_decay": float(model_config.get('weight_decay', 0.0)),
            "optimizer": str(model_config.get('optimizer', 'adam')),
            "momentum": float(model_config.get('momentum', 0.9))
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
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # 初始化训练历史记录器
    num_rounds = config.server['num_rounds']
    history = get_training_history()
    history.reset()  # 重置历史记录
    history.config_info = {
        "model_name": model_name,
        "strategy": strategy_name,
        "num_clients": config.client['num_clients'],
        "num_rounds": num_rounds,
        "fraction_fit": config.server['fraction_fit'],
        "local_epochs": config.client.get('local_epochs', 1),
        "batch_size": batch_size,
        "attack_enabled": config.attack.get('enabled', False),
        "attack_type": config.attack.get('type', None),
    }
    
    # 策略参数
    strategy_kwargs = {
        'fraction_fit': config.server['fraction_fit'],
        'fraction_evaluate': 0.0,  # 禁用客户端评估，只使用服务器评估
        'min_fit_clients': max(2, int(config.client['num_clients'] * config.server['fraction_fit'])),
        'min_evaluate_clients': 0,  # 不需要客户端评估
        # 设置可用客户端数：降低到30%以支持更好的并行执行
        'min_available_clients': config.server.get(
            'min_available_clients',
            max(2, int(config.client['num_clients'] * 0.3))
        ),
        'initial_parameters': ndarrays_to_parameters(get_weights(model)),
        'on_fit_config_fn': get_fit_config_fn(),
        'evaluate_fn': get_evaluate_fn(model, test_loader, num_rounds),
        # 'evaluate_metrics_aggregation_fn': weighted_average,  # 禁用客户端评估聚合
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
        
        # 创建服务器配置 (可选地从配置读取 round_timeout，避免无限等待)
        server_config = ServerConfig(
            num_rounds=config.server['num_rounds'],
            round_timeout=config.server.get('round_timeout', None)
        )
        
        # 返回 ServerAppComponents 对象
        return ServerAppComponents(
            config=server_config,
            strategy=strategy
        )
    
    return ServerApp(server_fn=server_fn)
