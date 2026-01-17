# 联邦学习简化项目

基于 Flower 框架的联邦学习实现，支持 MNIST、Fashion-MNIST 和 CIFAR-10 数据集。

## 📋 项目概述

这是一个简化的联邦学习项目，参考了 FedGreed 项目的实现逻辑，但使用标准的 Python 虚拟环境管理（不使用 Poetry）。项目实现了基本的联邦学习功能，包括：

- 多种数据集支持（MNIST、Fashion-MNIST、CIFAR-10）
- 多种聚合策略（FedAvg、FedMedian、FedTrimmedMean）
- IID 和 Non-IID 数据分区
- 基于 Flower 框架的仿真环境

## 🗂️ 项目结构

```
FederatedLearning-Simple/
├── src/                    # 核心源代码
│   ├── __init__.py
│   ├── models.py          # 模型定义（CNN模型）
│   ├── config.py          # 配置管理
│   ├── task.py            # 训练、测试和数据处理
│   ├── flower_client.py   # Flower客户端实现
│   ├── client_app.py      # 客户端应用
│   ├── server_app.py      # 服务器应用
│   └── strategies/        # 聚合策略
│       ├── __init__.py
│       └── fed_avg.py     # FedAvg、FedMedian、FedTrimmedMean
├── scripts/               # 运行脚本
│   ├── run_simulation.py  # 运行联邦学习仿真
│   └── partition_data.py  # 数据分区工具
├── configs/               # 配置文件
│   └── config.yaml        # 主配置文件
├── requirements.txt       # Python依赖
├── .gitignore
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

**要求：** Python 3.8+

创建虚拟环境（推荐）：

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行联邦学习仿真

直接运行默认配置：

```bash
python scripts/run_simulation.py
```

使用自定义配置：

```bash
python scripts/run_simulation.py --config configs/config.yaml
```

### 4. 数据分区（可选）

如果需要预先分区数据：

```bash
# IID分区
python scripts/partition_data.py MNIST --num_clients 10 --type iid

# Non-IID分区
python scripts/partition_data.py MNIST --num_clients 10 --type non_iid --alpha 0.5
```

## ⚙️ 配置说明

主配置文件：`configs/config.yaml`

```yaml
server:
  strategy: FedAvg          # 聚合策略: FedAvg, FedMedian, FedTrimmedMean
  fraction_fit: 1.0         # 每轮参与训练的客户端比例
  fraction_eval: 1.0        # 每轮参与评估的客户端比例
  num_rounds: 20            # 总轮数
  batch_size: 64            # 批量大小

client:
  num_clients: 10           # 客户端数量
  batch_size: 64            # 客户端批量大小
  local_epochs: 3           # 本地训练轮数

model:
  name: MNIST               # 模型: MNIST, FMNIST, CIFAR10
  learning_rate: 0.001      # 学习率

data:
  partitioning: iid         # 数据分区: iid 或 non_iid
  alpha: 0.5                # Dirichlet参数（仅用于non_iid）

backend:
  client_resources:
    num_cpus: 2.0           # 每个客户端的CPU资源
    num_gpus: 0.0           # 每个客户端的GPU资源
```

## 🎯 核心功能

### 1. 支持的数据集

- **MNIST**: 手写数字识别（28×28灰度图像）
- **Fashion-MNIST**: 时尚物品分类（28×28灰度图像）
- **CIFAR-10**: 自然图像分类（32×32彩色图像）

### 2. 聚合策略

- **FedAvg**: 联邦平均，标准的参数平均聚合
- **FedMedian**: 联邦中位数，使用中位数聚合参数（对异常值更鲁棒）
- **FedTrimmedMean**: 联邦修剪平均，移除极端值后平均

### 3. 数据分区

- **IID**: 独立同分布，数据随机均匀分配给客户端
- **Non-IID**: 非独立同分布，使用 Dirichlet 分布模拟数据异构性

## 📊 实验示例

### 示例1: MNIST + FedAvg (IID)

```yaml
# configs/config.yaml
server:
  strategy: FedAvg
  num_rounds: 10

model:
  name: MNIST

data:
  partitioning: iid
```

运行：
```bash
python scripts/run_simulation.py
```

### 示例2: CIFAR-10 + FedMedian (Non-IID)

```yaml
server:
  strategy: FedMedian
  num_rounds: 20

model:
  name: CIFAR10

data:
  partitioning: non_iid
  alpha: 0.1
```

## 🔧 自定义扩展

### 添加新的聚合策略

1. 在 `src/strategies/` 下创建新的策略文件
2. 继承 `FedAvgStrategy` 或 `flwr.server.strategy.Strategy`
3. 在 `src/server_app.py` 中注册新策略

```python
# src/strategies/my_strategy.py
from src.strategies.fed_avg import FedAvgStrategy

class MyCustomStrategy(FedAvgStrategy):
    def aggregate_fit(self, server_round, results, failures):
        # 自定义聚合逻辑
        pass
```

### 添加新的模型

在 `src/models.py` 中添加新模型：

```python
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # 定义网络层
    
    def forward(self, x):
        # 前向传播
        pass

MODELS['MyDataset'] = {
    'model': MyNet(),
    'num_classes': 10,
    'transforms': transforms.Compose([...])
}
```

## 📝 与原项目的差异

与 FedGreed 项目相比，本项目：

1. **不使用 Poetry**：使用标准的 `requirements.txt` 和虚拟环境
2. **简化结构**：移除了攻击模块和高级防御策略
3. **专注核心**：实现了基本的联邦学习流程和常用聚合策略
4. **易于理解**：代码注释详细，结构清晰

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

本项目参考了 [FedGreed](https://github.com/...) 的实现逻辑，使用了 [Flower](https://flower.dev/) 框架。
