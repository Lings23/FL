"""
诊断脚本 - 验证训练和评估是否正常工作
"""
import sys
sys.path.insert(0, '.')

import torch
from torch.utils.data import DataLoader, Subset
from src.models import CIFAR10NetV2, get_weights, set_weights
from src.task import train, test, load_datasets

print("=" * 80)
print("CIFAR10 训练和评估诊断")
print("=" * 80)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n使用设备: {device}")

# 加载数据
print("\n1. 加载CIFAR-10数据集...")
train_dataset, test_dataset = load_datasets('CIFAR10', './datasets')
train_subset = Subset(train_dataset, range(1000))
test_subset = Subset(test_dataset, range(500))

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

# 创建模型
print("\n2. 创建CIFAR10NetV2模型...")
model = CIFAR10NetV2()

# 测试初始性能
print("\n3. 测试初始性能（随机初始化）...")
initial_loss, initial_acc = test(model, test_loader, device)
print(f"   初始损失: {initial_loss:.4f}")
print(f"   初始准确率: {initial_acc:.2f}% (随机猜测约为10%)")

# 保存初始权重
initial_weights = get_weights(model)

# 训练模型
print("\n4. 训练模型（SGD, lr=0.01, 5 epochs）...")
train(model, train_loader, epochs=5, lr=0.01, device=device,
      weight_decay=0.0005, optimizer_type='sgd', momentum=0.9, client_id=0)

# 测试训练后性能
print("\n5. 测试训练后性能...")
trained_loss, trained_acc = test(model, test_loader, device)
print(f"   训练后损失: {trained_loss:.4f}")
print(f"   训练后准确率: {trained_acc:.2f}%")

# 验证权重是否改变
trained_weights = get_weights(model)
weight_changed = False
for i, (w1, w2) in enumerate(zip(initial_weights, trained_weights)):
    if not torch.allclose(torch.tensor(w1), torch.tensor(w2)):
        weight_changed = True
        break

print("\n6. 验证权重更新...")
print(f"   权重是否改变: {'✓ 是' if weight_changed else '✗ 否（问题！）'}")

# 测试权重设置和恢复
print("\n7. 测试权重设置功能...")
set_weights(model, initial_weights)
restored_loss, restored_acc = test(model, test_loader, device)
print(f"   恢复初始权重后准确率: {restored_acc:.2f}%")
print(f"   是否恢复到初始状态: {'✓ 是' if abs(restored_acc - initial_acc) < 1.0 else '✗ 否'}")

# 再次设置训练后的权重
set_weights(model, trained_weights)
final_loss, final_acc = test(model, test_loader, device)
print(f"   恢复训练权重后准确率: {final_acc:.2f}%")
print(f"   是否恢复到训练状态: {'✓ 是' if abs(final_acc - trained_acc) < 1.0 else '✗ 否'}")

# 总结
print("\n" + "=" * 80)
print("诊断结果总结")
print("=" * 80)
print(f"初始准确率:   {initial_acc:.2f}%")
print(f"训练后准确率: {trained_acc:.2f}%")
print(f"准确率提升:   {trained_acc - initial_acc:+.2f}%")

if trained_acc > initial_acc + 5:
    print("\n✓ 训练正常工作！模型在学习。")
elif trained_acc > initial_acc:
    print("\n⚠ 训练有轻微提升，但可能需要更多轮次。")
else:
    print("\n✗ 训练可能有问题！准确率没有提升。")
    print("   可能原因：")
    print("   1. 学习率设置不当")
    print("   2. 优化器配置问题")
    print("   3. 数据加载问题")

if weight_changed:
    print("✓ 权重更新正常")
else:
    print("✗ 权重没有更新（严重问题！）")

print("\n" + "=" * 80)
