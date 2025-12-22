import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from model import DeepCNN
from data_loader import get_train_loader, get_test_loader

# ================= 数据 =================
train_loader = get_train_loader(batch_size=64)
test_loader = get_test_loader(batch_size=64)

# ================= 模型/损失/优化器 =================
model = DeepCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_history = []
# ================= 训练 + 测试循环 =================
num_epochs = 15  # 先跑1个epoch测试，后续可以改
for epoch in range(num_epochs):
    model.train()  # 切换训练模式
    total_train = 0
    correct_train = 0

    for step, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        # 训练集准确率
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        train_acc = 100 * correct_train / total_train

        if step % 10 == 0:
            print(f"Epoch [{epoch+1}], Step [{step}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.2f}%")

    # ===== 测试集验证 =====
    model.eval()  # 切换评估模式
    total_test = 0
    correct_test = 0
    with torch.no_grad():  # 不计算梯度
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_acc = 100 * correct_test / total_test
    print(f"Epoch [{epoch+1}] Test Accuracy: {test_acc:.2f}%\n")

# 绘制损失函数变化
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.savefig('loss_curve.png') # 图片将保存在代码所在的当前文件夹下
print("Loss curve saved as loss_curve.png")

torch.save(model.state_dict(), 'cifar10_cnn.pth')
print("Model saved!")