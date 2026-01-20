# 攻击配置示例

本系统实现了动态攻击注入机制，可以通过修改 `configs/config.yaml` 文件来启用和配置不同的攻击类型，无需修改代码。

## 支持的攻击类型

### 1. 标签翻转攻击 (flip_labels)
将训练数据的标签进行翻转，例如在10类分类任务中，标签 y 会被翻转为 9-y。

**配置示例：**
```yaml
attack:
  enabled: true
  type: flip_labels
  malicious_ratio: 0.2  # 20%的客户端为恶意客户端
  params:
    num_classes: 10
```

### 2. 高斯噪声攻击 (gaussian_noise)
在模型梯度中添加高斯噪声，干扰模型收敛。

**配置示例：**
```yaml
attack:
  enabled: true
  type: gaussian_noise
  malicious_clients: [0, 2, 5]  # 指定恶意客户端ID
  params:
    mean: 0.0
    std: 1.0
```

### 3. 符号翻转攻击 (flip_sign)
将梯度的符号翻转，使模型朝相反方向更新。

**配置示例：**
```yaml
attack:
  enabled: true
  type: flip_sign
  malicious_ratio: 0.3
  params: {}
```

### 4. 梯度缩放攻击 (scale)
将梯度放大指定倍数，放大恶意更新的影响。

**配置示例：**
```yaml
attack:
  enabled: true
  type: scale
  malicious_ratio: 0.1
  params:
    scale_factor: 10.0
```

### 5. 零梯度攻击 (zero_gradient)
将梯度设置为零，使恶意客户端不贡献任何更新。

**配置示例：**
```yaml
attack:
  enabled: true
  type: zero_gradient
  malicious_ratio: 0.2
  params: {}
```

### 6. 随机更新攻击 (random_update)
用随机值替换梯度。

**配置示例：**
```yaml
attack:
  enabled: true
  type: random_update
  malicious_ratio: 0.2
  params: {}
```

## 配置说明

### enabled
- 类型：布尔值
- 说明：是否启用攻击
- 默认值：false

### type
- 类型：字符串
- 说明：攻击类型
- 可选值：flip_labels, gaussian_noise, flip_sign, scale, zero_gradient, random_update
- 默认值：null

### malicious_clients
- 类型：列表
- 说明：指定恶意客户端的ID列表
- 示例：[0, 2, 5]
- 默认值：[]

### malicious_ratio
- 类型：浮点数（0.0-1.0）
- 说明：恶意客户端占总客户端的比例（如果指定了malicious_clients，则忽略此项）
- 示例：0.2 表示 20% 的客户端为恶意
- 默认值：0.0

### params
- 类型：字典
- 说明：各攻击类型的特定参数
- 不同攻击类型需要不同的参数，见上面各攻击类型说明

## 使用方法

### 1. 基本使用

修改 `configs/config.yaml` 文件：

```yaml
# 联邦学习配置文件

server:
  strategy: FedAvg
  fraction_fit: 1.0
  fraction_eval: 1.0
  num_rounds: 20
  batch_size: 64

client:
  num_clients: 10
  batch_size: 64
  local_epochs: 3

model:
  name: MNIST
  learning_rate: 0.001

general:
  random_seed: 42

backend:
  client_resources:
    num_cpus: 2.0
    num_gpus: 0.0

data:
  partitioning: iid
  alpha: 0.5

# 启用攻击
attack:
  enabled: true
  type: flip_labels
  malicious_ratio: 0.2
  params:
    num_classes: 10
```

### 2. 运行仿真

```bash
# Windows
.\run.ps1

# Linux/Mac
./run.sh
```

### 3. 切换攻击类型

只需修改 `config.yaml` 中的 `type` 字段，无需修改任何代码：

```yaml
attack:
  enabled: true
  type: gaussian_noise  # 改为高斯噪声攻击
  malicious_ratio: 0.3
  params:
    mean: 0.0
    std: 2.0
```

### 4. 禁用攻击

```yaml
attack:
  enabled: false  # 设置为 false
  type: null
  malicious_clients: []
  malicious_ratio: 0.0
  params: {}
```

## 代码结构

```
src/
├── attacks.py           # 攻击函数实现
├── attack_manager.py    # 攻击管理器，负责动态加载和执行攻击
├── config.py           # 配置管理
├── client_app.py       # 客户端应用（集成攻击管理器）
├── flower_client.py    # Flower客户端（支持攻击注入）
└── task.py            # 训练任务（支持攻击执行）
```

## 扩展新攻击

要添加新的攻击类型：

1. 在 `src/attacks.py` 中添加新的攻击函数：

```python
def my_new_attack(parameters, param1, param2):
    """新攻击的实现"""
    for param in parameters:
        if param.grad is not None:
            # 实现攻击逻辑
            pass
```

2. 在 `ATTACK_REGISTRY` 中注册：

```python
ATTACK_REGISTRY = {
    # ...existing attacks...
    'my_new_attack': my_new_attack,
}
```

3. 在 `src/attack_manager.py` 的 `apply_attack` 方法中添加处理逻辑：

```python
elif self.attack_type == 'my_new_attack':
    param1 = self.params.get('param1', default_value)
    param2 = self.params.get('param2', default_value)
    attack_func(attack_target, param1, param2)
    return None
```

4. 在 `config.yaml` 中使用：

```yaml
attack:
  enabled: true
  type: my_new_attack
  malicious_ratio: 0.2
  params:
    param1: value1
    param2: value2
```

## 注意事项

1. **恶意客户端选择**：恶意客户端是随机选择的，每次运行可能不同。可以通过设置 `general.random_seed` 来保证可重复性。

2. **攻击类型互斥**：每次只能启用一种攻击类型。

3. **标签攻击 vs 梯度攻击**：
   - 标签攻击（flip_labels）：在训练前修改标签
   - 梯度攻击（其他类型）：在反向传播后修改梯度

4. **性能影响**：攻击会影响联邦学习的收敛速度和最终性能，这是预期行为。
