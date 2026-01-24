# 串行训练和准确率问题修复

## 问题现象

1. **串行训练问题**: 所有客户端使用相同的 pid，串行执行
2. **准确率停滞在10%**: CIFAR10 服务器评估准确率维持在 10%（随机猜测水平）

## 根本原因

### 问题1: 串行训练
- `min_available_clients` 设置过高（80%）
- 导致 Ray 等待所有客户端准备好，造成串行执行

### 问题2: 准确率不提升
可能原因：
1. 客户端评估（aggregate_evaluate）干扰训练
2. 服务器评估时模型设备不匹配
3. 训练参数配置不当（batch_size, epochs）

## 解决方案

### ✅ 修改 1: 禁用客户端评估

**src/server_app.py**
```python
strategy_kwargs = {
    'fraction_evaluate': 0.0,  # 禁用客户端评估
    'min_evaluate_clients': 0,
    # 'evaluate_metrics_aggregation_fn': weighted_average,  # 注释掉
}
```

**原因**: 客户端评估会：
- 增加通信开销
- 可能干扰训练流程
- 服务器评估已经足够

### ✅ 修改 2: 降低 min_available_clients

**src/server_app.py**
```python
'min_available_clients': max(2, int(config.client['num_clients'] * 0.3))
# 从 80% 降到 30%，10个客户端只需3个ready就开始
```

**效果**: 允许更多客户端并行执行

### ✅ 修改 3: 修复服务器评估

**src/server_app.py**
```python
def evaluate(server_round, parameters_ndarrays, config):
    set_weights(model, parameters_ndarrays)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 添加这行！确保模型在正确设备上
    loss, accuracy = test(model, test_loader, device)
    print(f"  设备: {device}")  # 添加设备信息
```

### ✅ 修改 4: 添加训练日志

**src/task.py**
```python
# 在训练循环中添加：
for epoch in range(epochs):
    epoch_loss = 0.0
    epoch_batches = 0
    
    for images, labels in train_loader:
        # ... 训练代码 ...
        epoch_loss += loss.item()
        epoch_batches += 1
    
    # 打印进度
    if epoch == 0 or epoch == epochs - 1:
        avg_loss = epoch_loss / epoch_batches
        print(f"[Client {client_id}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
```

### ✅ 修改 5: 调整配置参数

**configs/config.yaml**
```yaml
client:
  batch_size: 32  # 从64改为32，匹配CIFAR10配置
  local_epochs: 5  # 从3改为5，充分训练
```

## 验证步骤

### 1. 运行诊断脚本
```powershell
python diagnose_training.py
```

**预期输出**:
```
初始准确率:   10-15%
训练后准确率: 30-40%
准确率提升:   +20-30%
✓ 训练正常工作！模型在学习。
✓ 权重更新正常
```

### 2. 运行完整训练
```powershell
.\run.ps1
```

**观察日志**:
```
(ClientAppActor pid=7016) [Client 0] Epoch 1/5, Loss: 2.30
(ClientAppActor pid=7020) [Client 1] Epoch 1/5, Loss: 2.28  # 不同的pid ✓
(ClientAppActor pid=7024) [Client 2] Epoch 1/5, Loss: 2.31  # 不同的pid ✓

[服务器评估 - 轮次 1]
  损失: 2.25
  准确率: 15.32%  # 开始提升 ✓
  设备: cuda:0

[服务器评估 - 轮次 5]
  损失: 1.85
  准确率: 35.20%  # 继续提升 ✓
```

## 预期效果

### 并行训练
- ✅ 不同客户端使用不同的 pid
- ✅ 训练速度显著提升（接近线性加速）

### 准确率提升
| 轮次 | 预期准确率 |
|------|-----------|
| 1 | 15-20% |
| 5 | 30-40% |
| 10 | 45-55% |
| 20 | 60-65% |

## 故障排查

### 如果仍然串行执行
1. 检查 Ray 资源配置
   ```yaml
   backend:
     client_resources:
       num_cpus: 2.0
       num_gpus: 0.1  # 如果有GPU，确保<1.0
   ```

2. 在配置文件中显式设置
   ```yaml
   server:
     min_available_clients: 3
   ```

### 如果准确率仍不提升
1. 运行诊断脚本验证训练
2. 检查优化器配置（SGD vs Adam）
3. 确认数据加载正常
4. 查看训练日志中的 loss 是否下降

## 总结

| 问题 | 修复 | 位置 |
|------|------|------|
| 串行训练 | min_available_clients: 0.8→0.3 | server_app.py |
| 客户端评估干扰 | fraction_evaluate: 1.0→0.0 | server_app.py |
| 设备不匹配 | 添加 model.to(device) | server_app.py |
| 缺少训练日志 | 添加 epoch loss 打印 | task.py |
| 配置不匹配 | 调整 batch_size 和 epochs | config.yaml |

所有修改已完成，现在可以运行训练了！
