# 配置系统快速指南

## 🎯 核心理念

**不同模型需要不同的优化策略** - 本项目实现了模型特定配置系统，确保每个模型都使用最适合自己的超参数。

## 📊 模型配置对比

| 模型 | 参数量 | 优化器 | 学习率 | 权重衰减 | Batch Size | Epochs |
|------|--------|--------|--------|----------|-----------|--------|
| MNIST | 80K | Adam | 0.001 | 0.0 | 64 | 3 |
| FMNIST | 80K | Adam | 0.001 | 0.0 | 64 | 3 |
| CIFAR10 | 3.25M | **SGD** | **0.01** | **5e-4** | **32** | **5** |

**关键差异**: CIFAR10模型参数是MNIST的**40倍**，需要完全不同的训练策略。

## 🚀 快速开始

### 运行MNIST实验
```powershell
python scripts/run_simulation.py --config configs/config_mnist.yaml
```

### 运行Fashion-MNIST实验
```powershell
python scripts/run_simulation.py --config configs/config_fmnist.yaml
```

### 运行CIFAR-10实验（优化版）
```powershell
python scripts/run_simulation.py --config configs/config.yaml
```

## 🔧 配置文件说明

### config.yaml (默认配置 - CIFAR10)
```yaml
model:
  name: CIFAR10
  optimizer: null      # 使用model_configs中的sgd
  learning_rate: null  # 使用model_configs中的0.01
  weight_decay: null   # 使用model_configs中的5e-4

model_configs:
  CIFAR10:
    optimizer: sgd
    learning_rate: 0.01
    weight_decay: 5e-4
    momentum: 0.9
    batch_size: 32
    local_epochs: 5
```

### 覆盖特定参数
如果想临时修改学习率，直接在model字段中设置：
```yaml
model:
  name: CIFAR10
  learning_rate: 0.005  # 覆盖默认的0.01
  optimizer: null       # 仍使用默认的sgd
```

## ✅ 验证测试

### 测试配置系统
```powershell
python test_config.py
```

### 验证模型独立性
```powershell
python verify_config_independence.py
```

## 📈 预期效果

### MNIST/FMNIST
- **配置**: Adam, lr=0.001
- **20轮准确率**: 97-99%
- **状态**: ✅ 配置稳定，效果正常

### CIFAR10（旧配置）
- **配置**: Adam, lr=0.001
- **20轮准确率**: 40-50%
- **状态**: ❌ 效果差，已优化

### CIFAR10（新配置）
- **配置**: SGD, lr=0.01, weight_decay=5e-4
- **20轮准确率**: 60-65%
- **50轮准确率**: 75-80%
- **状态**: ✅ 大幅改进

## 🎓 为什么这样配置？

### MNIST/FMNIST使用Adam
- ✅ 参数少（8万），容易训练
- ✅ Adam自适应学习率，收敛快
- ✅ 小学习率足够稳定

### CIFAR10使用SGD
- ✅ 参数多（325万），需要强正则化
- ✅ SGD泛化能力更好
- ✅ 更大学习率才能有效更新大模型
- ✅ Weight Decay防止过拟合

## 📚 详细文档

- [CIFAR10优化详解](CIFAR10_OPTIMIZATION.md)
- [配置系统架构](CONFIG_SYSTEM.md)
- [攻击配置说明](ATTACK_CONFIG.md)

## 🔍 常见问题

**Q: 修改CIFAR10配置会影响MNIST吗？**  
A: 不会！配置系统确保每个模型使用独立的参数。

**Q: 如何添加新模型？**  
A: 在`model_configs`中添加新模型的配置即可。

**Q: 可以强制所有模型使用相同配置吗？**  
A: 可以，在`model`字段中设置非null值会覆盖所有模型的配置。

**Q: 如何恢复默认配置？**  
A: 将`model`字段中的参数改为`null`即可使用model_configs中的默认值。

## 🎉 开始实验

一切就绪！现在可以放心地：
1. ✅ 运行MNIST实验 - 使用适合的Adam优化器
2. ✅ 运行FMNIST实验 - 使用适合的Adam优化器  
3. ✅ 运行CIFAR10实验 - 使用优化的SGD配置

各模型互不影响，都能达到最佳效果！
