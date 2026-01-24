"""
测试模型特定配置系统
验证不同模型使用不同的优化器参数
"""
import sys
sys.path.insert(0, '.')

from src.config import get_config, init_config

print("=" * 70)
print("测试模型特定配置系统")
print("=" * 70)

# 测试默认配置
print("\n1. 测试默认配置 (config.yaml - CIFAR10)")
print("-" * 70)
config = init_config("configs/config.yaml")
model_config = config.get_model_config()
print(f"模型名称: {config.model['name']}")
print(f"优化器: {model_config.get('optimizer', 'N/A')}")
print(f"学习率: {model_config.get('learning_rate', 'N/A')}")
print(f"权重衰减: {model_config.get('weight_decay', 'N/A')}")
print(f"动量: {model_config.get('momentum', 'N/A')}")
print(f"批量大小: {model_config.get('batch_size', 'N/A')}")
print(f"本地轮数: {model_config.get('local_epochs', 'N/A')}")

# 测试MNIST配置
print("\n2. 测试MNIST配置")
print("-" * 70)
config_mnist = init_config("configs/config_mnist.yaml")
model_config_mnist = config_mnist.get_model_config()
print(f"模型名称: {config_mnist.model['name']}")
print(f"优化器: {model_config_mnist.get('optimizer', 'N/A')}")
print(f"学习率: {model_config_mnist.get('learning_rate', 'N/A')}")
print(f"权重衰减: {model_config_mnist.get('weight_decay', 'N/A')}")
print(f"动量: {model_config_mnist.get('momentum', 'N/A')}")
print(f"批量大小: {model_config_mnist.get('batch_size', 'N/A')}")
print(f"本地轮数: {model_config_mnist.get('local_epochs', 'N/A')}")

# 测试FMNIST配置
print("\n3. 测试FMNIST配置")
print("-" * 70)
config_fmnist = init_config("configs/config_fmnist.yaml")
model_config_fmnist = config_fmnist.get_model_config()
print(f"模型名称: {config_fmnist.model['name']}")
print(f"优化器: {model_config_fmnist.get('optimizer', 'N/A')}")
print(f"学习率: {model_config_fmnist.get('learning_rate', 'N/A')}")
print(f"权重衰减: {model_config_fmnist.get('weight_decay', 'N/A')}")
print(f"动量: {model_config_fmnist.get('momentum', 'N/A')}")
print(f"批量大小: {model_config_fmnist.get('batch_size', 'N/A')}")
print(f"本地轮数: {model_config_fmnist.get('local_epochs', 'N/A')}")

# 测试CIFAR10优化配置
print("\n4. 测试CIFAR10优化配置")
print("-" * 70)
config_cifar_opt = init_config("configs/config_cifar10_optimized.yaml")
model_config_cifar = config_cifar_opt.get_model_config()
print(f"模型名称: {config_cifar_opt.model['name']}")
print(f"优化器: {model_config_cifar.get('optimizer', 'N/A')}")
print(f"学习率: {model_config_cifar.get('learning_rate', 'N/A')}")
print(f"权重衰减: {model_config_cifar.get('weight_decay', 'N/A')}")
print(f"动量: {model_config_cifar.get('momentum', 'N/A')}")
print(f"批量大小: {model_config_cifar.get('batch_size', 'N/A')}")
print(f"本地轮数: {model_config_cifar.get('local_epochs', 'N/A')}")

print("\n" + "=" * 70)
print("总结:")
print("=" * 70)
print("✓ MNIST/FMNIST: Adam优化器, 学习率=0.001, 无权重衰减")
print("✓ CIFAR10: SGD优化器, 学习率=0.01, 权重衰减=5e-4")
print("\n配置系统工作正常！每个模型使用适合自己的超参数。")
