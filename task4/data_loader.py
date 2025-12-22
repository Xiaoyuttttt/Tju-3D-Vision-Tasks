from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_train_loader(batch_size=64):
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转10度
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader


def get_test_loader(batch_size=64):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,      # 注意这里是 False
        download=True,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False     # 测试集通常不打乱
    )

    return test_loader
