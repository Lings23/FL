# CIFAR10NetV2 性能分析与优化方案

## 问题诊断

您的CIFAR10NetV2模型准确率停留在40%-50%的主要原因：

### 1. **学习率过低** ⚠️
- **当前值**: 0.001
- **问题**: CIFAR10NetV2有325万参数，比原始CIFAR10Net深得多，需要更大的学习率才能有效更新权重
- **建议**: 提高到0.01（提高10倍）

### 2. **优化器选择不当** ⚠️
- **当前**: Adam优化器，无权重衰减
- **问题**: Adam对CIFAR-10这类图像分类任务，容易在初期快速下降后停滞
- **建议**: 使用SGD + Momentum (0.9) + Weight Decay (5e-4)

### 3. **训练不充分** ⚠️
- **当前**: 每个客户端只训练3个epoch
- **问题**: 对于深度网络，3个epoch根本不够让模型收敛
- **建议**: 增加到5个epoch

### 4. **缺少正则化** ⚠️
- **当前**: 无权重衰减
- **问题**: 325万参数的模型很容易过拟合
- **建议**: 添加L2正则化 (weight_decay=5e-4)

### 5. **总训练轮数不足** ⚠️
- **当前**: 20轮联邦学习
- **问题**: CIFAR-10通常需要150-200个epoch才能达到良好性能
- **建议**: 增加到50轮（相当于250个epoch的分布式训练）

### 6. **批量大小偏大** ⚠️
- **当前**: batch_size=64
- **问题**: 较大的batch size会降低梯度估计的随机性，影响泛化
- **建议**: 减小到32

## 已实施的改进

### 代码层面修改

1. **task.py**: 
   - 将Adam优化器替换为SGD + Momentum
   - 添加weight_decay参数支持
   ```python
   optimizer = torch.optim.SGD(
       model.parameters(), 
       lr=lr, 
       momentum=0.9, 
       weight_decay=weight_decay
   )
   ```

2. **config.yaml**: 
   - learning_rate: 0.001 → 0.01
   - local_epochs: 3 → 5
   - batch_size: 64 → 32
   - 添加weight_decay: 5e-4

3. **flower_client.py & server_app.py**: 
   - 传递weight_decay参数到训练函数

## 使用方法

### 方式1：使用优化后的默认配置
直接运行，已更新的config.yaml包含优化参数：
```powershell
.\run.ps1
```

### 方式2：使用专门的CIFAR10优化配置
```powershell
# PowerShell
python scripts/run_simulation.py --config configs/config_cifar10_optimized.yaml

# 或使用环境变量
$env:FL_CONFIG="configs/config_cifar10_optimized.yaml"
.\run.ps1
```

## 预期效果

使用优化后的配置，您应该看到：

- **前10轮**: 准确率从10%左右快速提升到60-65%
- **10-30轮**: 稳步提升到70-75%
- **30-50轮**: 逐步提升到75-80%

如果使用GPU训练，效果会更好，可达80-85%。

## 进一步优化建议

如果仍需提升性能，可以考虑：

1. **学习率调度**: 
   - 每15轮将学习率减半
   - 在后期使用更小的学习率精调

2. **数据增强**（已在models.py中实现）:
   - RandomCrop(32, padding=4)
   - RandomHorizontalFlip()

3. **更深的网络架构**:
   - 添加残差连接（ResNet风格）
   - 使用更多卷积层

4. **调整联邦学习参数**:
   - 如果是non-iid数据，尝试降低alpha值（如0.1）
   - 增加客户端参与比例

## 对比实验

| 配置 | 学习率 | 优化器 | Local Epochs | 预期准确率 |
|------|--------|--------|--------------|-----------|
| 原配置 | 0.001 | Adam | 3 | 40-50% |
| 优化配置 | 0.01 | SGD+Momentum | 5 | 75-80% |

## 问题排查

如果优化后仍然效果不佳：

1. **检查数据加载**: 确认CIFAR-10数据集正确下载和加载
2. **验证归一化**: 确认使用了正确的均值和标准差
3. **监控训练过程**: 查看每轮的loss是否稳定下降
4. **检查梯度**: 确认梯度没有爆炸或消失

## 参考

CIFAR-10最佳实践：
- 标准SGD训练200 epochs可达85-90%
- 联邦学习由于数据分散，通常需要更多轮次
- 深度网络需要较大学习率和适当正则化
