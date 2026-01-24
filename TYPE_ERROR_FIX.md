# 类型错误修复说明

## 问题描述

运行联邦学习时出现错误：
```
TypeError: '<=' not supported between instances of 'float' and 'str'
```

## 根本原因

**YAML配置文件中的科学计数法被某些解析器读取为字符串**

在配置文件中：
```yaml
weight_decay: 5e-4  # ❌ 可能被读作字符串 "5e-4"
```

当传递给PyTorch优化器时：
```python
torch.optim.Adam(..., weight_decay="5e-4")  # ❌ 字符串无法比较
```

## 修复方案

### 1. 配置文件修改（治本）

将科学计数法改为标准小数形式：
```yaml
# 修改前
weight_decay: 5e-4

# 修改后
weight_decay: 0.0005  # 5e-4
```

**已修复的文件：**
- ✅ `configs/config.yaml`
- ✅ `configs/config_mnist.yaml`
- ✅ `configs/config_fmnist.yaml`
- ✅ `configs/config_cifar10_optimized.yaml`

### 2. 代码防御（治标）

在使用配置值前添加类型转换：

**修改位置1: src/server_app.py**
```python
# 修改前
return {
    "lr": model_config.get('learning_rate', 0.001),
    "weight_decay": model_config.get('weight_decay', 0.0),
    ...
}

# 修改后
return {
    "lr": float(model_config.get('learning_rate', 0.001)),
    "weight_decay": float(model_config.get('weight_decay', 0.0)),
    ...
}
```

**修改位置2: src/task.py**
```python
# 在train函数开始处添加
def train(...):
    # 确保参数类型正确（防止YAML解析问题）
    lr = float(lr)
    weight_decay = float(weight_decay)
    momentum = float(momentum)
    epochs = int(epochs)
```

## 验证修复

运行测试验证：
```powershell
python test_config.py
```

预期输出：
```
weight_decay值: 0.0005
weight_decay类型: <class 'float'>  ✓
```

## 为什么会发生这个问题？

### YAML规范中的数字类型

YAML标准支持科学计数法，但不同的解析器实现不一致：

| 格式 | Python yaml.safe_load() | 某些解析器 |
|------|------------------------|-----------|
| `5e-4` | ✓ float | ❌ str |
| `5.0e-4` | ✓ float | ✓ float |
| `0.0005` | ✓ float | ✓ float |

**最佳实践：**使用标准小数形式（`0.0005`）避免解析歧义。

## 其他常见YAML类型问题

```yaml
# ❌ 可能出问题的写法
num_gpus: 0.0  # 可能被读作字符串 "0.0"
learning_rate: 1e-3  # 可能被读作字符串

# ✓ 推荐写法
num_gpus: !!float 0.0  # 明确指定类型
learning_rate: 0.001  # 使用标准小数
```

## 总结

- ✅ **配置文件已修复** - 所有科学计数法改为标准小数
- ✅ **代码已加固** - 添加类型转换防止未来问题
- ✅ **问题已解决** - 可以正常运行训练

**建议：** 在所有YAML配置中避免使用科学计数法，统一使用标准小数形式。
