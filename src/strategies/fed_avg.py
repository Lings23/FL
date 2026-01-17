"""
联邦平均策略
实现标准的FedAvg聚合策略
"""
from typing import List, Tuple, Optional, Dict
from flwr.server.strategy import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy

from src.models import get_weights, set_weights, get_model
from src.task import test, save_model, save_results
from src.config import get_config


class FedAvgStrategy(FedAvg):
    """
    联邦平均策略
    
    继承自Flower的FedAvg，添加了结果保存和模型检查点功能
    """
    
    def __init__(self, *args, **kwargs):
        """初始化策略"""
        self.save_path = kwargs.pop('save_path', 'outputs')
        self.model = kwargs.pop('model', None)
        super().__init__(*args, **kwargs)
        
        # 跟踪最佳准确率
        self.best_accuracy = 0.0
        # 存储结果
        self.results = {
            'rounds': [],
            'centralized_accuracy': [],
            'centralized_loss': []
        }
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        聚合客户端的训练结果
        
        Args:
            server_round: 当前轮次
            results: 客户端训练结果列表
            failures: 失败列表
            
        Returns:
            (aggregated_parameters, metrics): 聚合后的参数和指标
        """
        # 调用父类的聚合方法
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            print(f"\n[轮次 {server_round}] 成功聚合 {len(results)} 个客户端的参数")
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        聚合客户端的评估结果
        
        Args:
            server_round: 当前轮次
            results: 客户端评估结果列表
            failures: 失败列表
            
        Returns:
            (aggregated_loss, metrics): 聚合后的损失和指标
        """
        # 调用父类的聚合方法
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # 打印评估结果
        if aggregated_metrics:
            print(f"[轮次 {server_round}] 聚合评估结果:")
            for key, value in aggregated_metrics.items():
                print(f"  - {key}: {value:.4f}")
        
        return aggregated_loss, aggregated_metrics


class FedMedianStrategy(FedAvgStrategy):
    """
    联邦中位数策略
    
    使用中位数而非平均值进行参数聚合，对异常值更鲁棒
    """
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """使用中位数聚合参数"""
        if not results:
            return None, {}
        
        import numpy as np
        
        # 提取所有客户端的参数
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # 计算中位数
        params_list = [weights for weights, _ in weights_results]
        
        # 对每一层参数计算中位数
        aggregated_params = []
        for layer_params in zip(*params_list):
            # 沿第0轴（客户端维度）计算中位数
            median_param = np.median(np.array(layer_params), axis=0)
            aggregated_params.append(median_param)
        
        print(f"\n[轮次 {server_round}] 使用中位数聚合 {len(results)} 个客户端的参数")
        
        return ndarrays_to_parameters(aggregated_params), {}


class FedTrimmedMeanStrategy(FedAvgStrategy):
    """
    联邦修剪平均策略
    
    移除最大和最小的beta比例的参数，然后计算平均值
    """
    
    def __init__(self, *args, beta: float = 0.1, **kwargs):
        """
        初始化
        
        Args:
            beta: 要修剪的比例（从两端各修剪beta比例）
        """
        super().__init__(*args, **kwargs)
        self.beta = beta
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """使用修剪平均聚合参数"""
        if not results:
            return None, {}
        
        import numpy as np
        
        # 提取所有客户端的参数
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        params_list = [weights for weights, _ in weights_results]
        num_clients = len(params_list)
        num_trim = int(num_clients * self.beta)
        
        # 对每一层参数进行修剪平均
        aggregated_params = []
        for layer_params in zip(*params_list):
            layer_array = np.array(layer_params)
            
            # 对每个参数值进行排序和修剪
            sorted_indices = np.argsort(layer_array, axis=0)
            
            # 修剪最小和最大的beta比例
            if num_trim > 0:
                # 创建掩码，保留中间的值
                mask = np.ones(num_clients, dtype=bool)
                mask[:num_trim] = False
                mask[-num_trim:] = False
                
                # 计算修剪后的平均值
                trimmed_params = layer_array[mask]
                mean_param = np.mean(trimmed_params, axis=0)
            else:
                mean_param = np.mean(layer_array, axis=0)
            
            aggregated_params.append(mean_param)
        
        print(f"\n[轮次 {server_round}] 使用修剪平均聚合 {len(results)} 个客户端的参数 (beta={self.beta})")
        
        return ndarrays_to_parameters(aggregated_params), {}
