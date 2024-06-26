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
from torchsummary import summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
class ResNet152(nn.Module):
    def __init__(self):
        super(ResNet152, self).__init__()
        self.model = models.resnet152(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 200)

    def forward(self, x):
        return self.model(x)


model = ResNet152().to(device)
print(summary(model, input_size=(3, 32, 32)))

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

    def training(train_loader):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
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
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        train_loss = total_loss / len(train_loader.dataset)
        train_accuracy = 100. * correct / total
        print(f"Train accuracy : {train_accuracy:.3f}%")
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)

        if epoch+1 == n_epochs:
            cm_test = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(85, 85))
            disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.arange(cm_test.shape[0]))
            disp_test.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
            plt.title('Model Training Confusion Matrix', fontsize=55)
            plt.xlabel('Predicted Label', fontsize=35)
            plt.ylabel('True Label', fontsize=35)
            plt.savefig('Model_Training_confusion_matrix.png', dpi=450, bbox_inches='tight')
            plt.show()
            
        return train_loss, train_accuracy

    def validation(valid_loader):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        valid_loss = val_loss / len(valid_loader.dataset)
        valid_accuracy = 100. * correct / total
        print(f"Validation accuracy : {valid_accuracy:.3f}%")
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_accuracy)

        if epoch+1 == n_epochs:
            cm_test = confusion_matrix(all_labels, all_preds)
            fig, ax = plt.subplots(figsize=(85, 85))
            disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.arange(cm_test.shape[0]))
            disp_test.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
            plt.title('Model Validation Confusion Matrix', fontsize=55)
            plt.xlabel('Predicted Label', fontsize=35)
            plt.ylabel('True Label', fontsize=35)
            plt.savefig('Model_Validation_confusion_matrix.png', dpi=450, bbox_inches='tight')
            plt.show()
            
        return valid_loss, valid_accuracy

    train_loss, train_acc = training(train_loader)
    val_loss, val_acc = validation(valid_loader)

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
