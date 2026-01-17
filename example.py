"""
简单示例：运行MNIST联邦学习
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import init_config

# 初始化配置
config = init_config("configs/config.yaml")

print("联邦学习示例")
print(f"模型: {config.model['name']}")
print(f"客户端数量: {config.client['num_clients']}")
print(f"训练轮数: {config.server['num_rounds']}")

# 运行仿真
from scripts.run_simulation import main
main()
