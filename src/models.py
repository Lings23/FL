"""
模型定义模块
定义了MNIST、Fashion-MNIST和CIFAR-10的CNN模型
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class MNISTNet(nn.Module):
    """MNIST/Fashion-MNIST 的简单CNN模型 (28x28 单通道图像)"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)  # 输入1通道，输出16通道，5x5卷积核 -> 16x24x24
        self.pool = nn.MaxPool2d(2, 2)    # 2x2池化 -> 16x12x12
        self.conv2 = nn.Conv2d(16, 32, 5) # 输入16通道，输出32通道 -> 32x8x8
        self.fc1 = nn.Linear(32 * 4 * 4, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)          # 输出层 (10类)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CIFAR10Net(nn.Module):
    """CIFAR-10 的CNN模型 (32x32 RGB图像)"""
    
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 输入3通道，输出32通道
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# 模型配置字典
MODELS = {
    'MNIST': {
        'model': MNISTNet(),
        'num_classes': 10,
        'transforms': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    },
    'FMNIST': {
        'model': MNISTNet(),
        'num_classes': 10,
        'transforms': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    },
    'CIFAR10': {
        'model': CIFAR10Net(),
        'num_classes': 10,
        'transforms': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
}


def get_weights(model: nn.Module):
    """从模型中提取权重为NumPy数组列表"""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights):
    """将权重设置到模型中"""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_model(model_name: str) -> nn.Module:
    """获取指定名称的模型"""
    if model_name not in MODELS:
        raise ValueError(f"未知的模型: {model_name}. 支持的模型: {list(MODELS.keys())}")
    return MODELS[model_name]['model']
