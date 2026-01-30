"""
联邦学习仿真运行脚本
使用Flower框架运行联邦学习仿真
"""
import sys
import os
from pathlib import Path
os.environ["APPDATA"] = "D:/python_appdata"
os.environ["LOCALAPPDATA"] = "D:/python_local"
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

from flwr.simulation import run_simulation

from src.config import init_config
from src.client_app import get_client_fn
from src.server_app import get_server_fn, get_training_history


def main(config_path: str = "configs/config.yaml"):
    """
    运行联邦学习仿真
    
    Args:
        config_path: 配置文件路径
    """
    # 初始化配置
    config = init_config(config_path)
    
    print("=" * 60)
    print("联邦学习仿真开始")
    print("=" * 60)
    print(f"\n配置信息:")
    print(f"  模型: {config.model['name']}")
    print(f"  客户端数量: {config.client['num_clients']}")
    print(f"  总轮数: {config.server['num_rounds']}")
    print(f"  聚合策略: {config.server.get('strategy', 'FedAvg')}")
    print(f"  数据分区: {config.data.get('partitioning', 'iid')}")
    print()
    
    # 创建输出目录
    os.makedirs("outputs", exist_ok=True)
    
    # 运行仿真并保存结果
    try:
        # 注意: run_simulation() 在新版 Flower 中返回 None
        # 真正的结果通过 TrainingHistory 收集
        run_simulation(
            server_app=get_server_fn(),
            client_app=get_client_fn(),
            num_supernodes=config.client['num_clients'],
            backend_config=config.backend
        )

        # 从 TrainingHistory 获取实际结果
        import json
        import datetime

        history = get_training_history()
        results = history.to_dict()
        
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.model.get('name', 'results')
        json_path = out_dir / f"results_{model_name}_{ts}.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {json_path}")
        
        # 打印训练摘要
        if results.get('accuracies') or results.get('centralized_accuracy'):
            accs = results.get('centralized_accuracy', [])
            if accs:
                print(f"\n训练摘要:")
                print(f"  最终准确率: {accs[-1]:.2f}%")
                print(f"  最佳准确率: {max(accs):.2f}%")

        print("\n" + "=" * 60)
        print("联邦学习仿真完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行联邦学习仿真")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    main(args.config)
