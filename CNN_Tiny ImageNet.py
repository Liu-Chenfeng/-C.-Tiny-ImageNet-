import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import trange
import copy
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim
import torch.nn.functional as F
import time
from torchsummary import summary


if torch.cuda.is_available():
    print('CUDA is available. Working on GPU')
    device = torch.device('cuda')
else:
    print('CUDA is not available. Working on CPU')
    device = torch.device('cpu')

DIR_MAIN = 'C:\\Users\\lcf14\\Desktop\\tiny-imagenet-200\\'
DIR_TRAIN = DIR_MAIN + 'train\\'
DIR_VAL = DIR_MAIN + 'val\\'
DIR_TEST = DIR_MAIN + 'test\\'

# labels的数量 - 200
labels = os.listdir(DIR_TRAIN)

# 初始化标签编码器    Инициализируйте кодировщик этикеток
encoder_labels = LabelEncoder()
encoder_labels.fit(labels)

# 创建训练文件和标签的列表    Создавайте списки файлов и меток для обучения (100'000 items)
files_train = []
labels_train = []
for label in labels:
    for filename in os.listdir(DIR_TRAIN + label + '\\images\\'):
        files_train.append(DIR_TRAIN + label + '\\images\\' + filename)
        labels_train.append(label)

# 创建验证文件和标签的列表    Создание списков файлов и меток для проверки (10'000 items)
files_val = []
labels_val = []
for filename in os.listdir(DIR_VAL + 'images\\'):
    files_val.append(DIR_VAL + 'images\\' + filename)

val_df = pd.read_csv(DIR_VAL + 'val_annotations.txt', sep='\t', names=["File", "Label", "X1", "Y1", "X2", "Y2"],
                     usecols=["File", "Label"])
for f in files_val:
    l = val_df.loc[val_df['File'] == f[len(DIR_VAL + 'images\\'):]]['Label'].values[0]
    labels_val.append(l)

# 创建测试文件和标签的列表    Создание списков файлов для тестирования (10'000 items)
files_test = []
for filename in os.listdir(DIR_TEST + 'images\\'):
    files_test.append(DIR_TEST + 'images\\' + filename)
    files_test = sorted(files_test)

"""
print("The first five files from the list of train images:", files_train[:5])
print("\nThe first five labels from the list of train labels:", labels_train[:5])
print("\nThe first five files from the list of validation images:", files_val[:5])
print("\nThe first five labels from the list of validation labels:", labels_val[:5])
print("\nThe first five files from the list of test images:", files_test[:5])
"""


class ImagesDataset(Dataset):
    def __init__(self, files, labels, encoder, transforms, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert('RGB')

        if self.mode == 'train' or self.mode == 'val':
            x = self.transforms(pic)
            label = self.labels[index]
            y = self.encoder.transform([label])[0]
            return x, y
        elif self.mode == 'test':
            x = self.transforms(pic)
            return x, self.files[index]


# 图像归一化    Нормализовать данные
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.06, 0.08), ratio=(1, 3), value=0, inplace=True)
])

transforms_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 初始化三个数据集（训练、验证和测试）
train_dataset = ImagesDataset(files=files_train,
                              labels=labels_train,
                              encoder=encoder_labels,
                              transforms=transforms_train,
                              mode='train')

val_dataset = ImagesDataset(files=files_val,
                            labels=labels_val,
                            encoder=encoder_labels,
                            transforms=transforms_val,
                            mode='val')

test_dataset = ImagesDataset(files=files_test,
                             labels=None,
                             encoder=None,
                             transforms=transforms_val,
                             mode='test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # First block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Output: (32, 112, 112)
            nn.Dropout(0.3)
        )

        # Second block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Output: (64, 56, 56)
            nn.Dropout(0.3)
        )

        # Third block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Output: (128, 28, 28)
            nn.Dropout(0.5)
        )

        # Fourth block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> MaxPool -> Dropout
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Output: (256, 14, 14)
            nn.Dropout(0.5)
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(14*14*256, 200)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.flatten(x)
        #print("After flatten:", x.shape)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


model = Net().to(device)
print(summary(model, input_size=(3, 224, 224)))

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)


n_epochs = 15
train_loss_list = []
valid_loss_list = []
test_loss_list = []
train_acc_list = []
valid_acc_list = []
test_acc_list = []


for epoch in range(n_epochs):
    print(f'EPOCH {epoch + 1}:')
    start_time = time.time()

    def training(train_loader):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = 100. * correct / total
        print(f"Train accuracy : {train_accuracy:.3f}%")
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        return train_loss, train_accuracy


    def validation(valid_loader):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        valid_loss = val_loss / len(valid_loader.dataset)
        valid_accuracy = 100. * correct / total
        print(f"Validation accuracy : {valid_accuracy:.3f}%")
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_accuracy)
        return valid_loss, valid_accuracy


    train_loss, train_acc = training(train_loader)
    val_loss, val_acc = validation(valid_loader)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch running time: {epoch_time} seconds")
print("Complete!")

# 绘制学习曲线
plt.figure(figsize=(15, 6))

# 绘制训练与验证损失
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss', color='#8502d1')
plt.plot(valid_loss_list, label='Validation Loss', color='darkorange')
plt.legend()
plt.title('Loss Evolution')

# 绘制训练与验证准确率
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='Train Accuracy', color='#8502d1')
plt.plot(valid_acc_list, label='Validation Accuracy', color='darkorange')
plt.legend()
plt.title('Accuracy Evolution')

plt.show()
