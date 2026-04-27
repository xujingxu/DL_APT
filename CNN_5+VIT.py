import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.optim as optim
from PIL import Image
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torchvision.models import shufflenet_v2_x2_0
from torchvision.models import vit_b_32
from torchvision.models import mobilenet_v3_large

def default_loader(path):
    return Image.open(path).convert('RGB')

### 模型输出是png图像路径 空格 label
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除本行string 字符串末尾的指定字符
            words = line.split()  # 用split将该行分割成列表
            imgs.append((words[0], int(float(words[1]))))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)  # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
        return img, label

    def __len__(self):
        return len(self.imgs)

# 图像的初始化操作
train_trainsforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomRotation((-30, 30)),
                                        transforms.ToTensor(), ])
test_trainsforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(), ])

# 数据集加载方式设置
dir = '/txt文件路径/'
train_data = MyDataset(txt=dir+'train.txt', transform=train_trainsforms)
val_data = MyDataset(txt=dir+'val.txt', transform=test_trainsforms)
test_data = MyDataset(txt=dir+'test.txt', transform=test_trainsforms)

train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=True)

# 定义超参数
epochs = 100
pre_epoch = 0
steps = 0
running_loss = 0
train_losses, val_losses = [], []
device_ids = [0]

##### resnet
# net = models.resnet34(pretrained=True)
# fc_features = net.fc.in_features
# net.fc = nn.Linear(fc_features, 3)  # 3需要改成自己类别数
# net = torch.nn.DataParallel(net, device_ids=[0])  # 声明所有可用设备
# model = net.cuda(device=device_ids[0])  # 模型放在主设备

##### shufflenet
# net = shufflenet_v2_x2_0(pretrained=True)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(in_features=num_ftrs, out_features=3, bias=True)
# net = torch.nn.DataParallel(net, device_ids=[0])  # 声明所有可用设备
# model = net.cuda(device=device_ids[0])  # 模型放在主设备

##### mobilenet
# net = mobilenet_v3_large(pretrained=True)
# in_features = net.classifier[3].in_features
# net.classifier[3] = nn.Linear(in_features, 3)
# model = net.cuda(device=device_ids[0])  # 模型放在主设备


##### VGG
# net = models.vgg16(pretrained=True)
# num_fc = net.classifier[6].in_features
# net.classifier[6] = torch.nn.Linear(num_fc,3)
# model = net.cuda(device=device_ids[0])  # 模型放在主设备


##### Alexnet
# net=models.AlexNet()
# net.classifier[6]=nn.Linear(2048,2)
# model = net.cuda(device=device_ids[0])  # 模型放在主设备



##### VIT
# net = vit_b_32(pretrained=True)
# num_ftrs = net.heads.head.in_features
# net.fc = nn.Linear(in_features=num_ftrs, out_features=2, bias=True)
# net = torch.nn.DataParallel(net, device_ids=[0])  # 声明所有可用设备
# model = net.cuda(device=device_ids[0])  # 模型放在主设备



lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

# 训练
if __name__ == "__main__":
    best_acc = 0.75
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    print("Start Training")  # 定义遍历数据集的次数
    with open("../acc_re.txt", "w") as f:
        with open("../log_re.txt", "w")as f2:
            for epoch in range(pre_epoch, epochs):
                print('\nEpoch: %d' % (epoch + 1))
                prob_all = []
                label_all = []
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_loader, 0):
                    # 准备数据
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device=device_ids[0]), labels.to(device=device_ids[0])
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    prob_all.extend(outputs.data[:, 1].cpu().numpy())
                    label_all.extend(labels.cpu().numpy())
                    auc = roc_auc_score(label_all, prob_all)
                    loss = sum_loss / (i + 1)
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | AUC: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, 100. * roc_auc_score(label_all, prob_all)))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% | AUC: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total, 100. * roc_auc_score(label_all, prob_all)))
                    f2.write('\n')
                    f2.flush()

                train_acc_list.append((correct / total).item())
                train_loss_list.append(loss)

                # 每训练完一个epoch测试一下准确率
                print("Waiting Val!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in val_loader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(
                            device=device_ids[0]), labels.to(device=device_ids[0])
                        outputs = model(images)

                        loss = criterion(outputs, labels)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('验证分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    # torch.save(net.state_dict(), './net_%03d.pth' % (epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" %
                            (epoch + 1, 100. * acc))
                    f.write('\n')
                    f.flush()
                    val_acc_list.append(acc.item())
                    val_loss_list.append(loss.item())
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("../best_acc_01.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" %
                                 (epoch + 1, 100. * acc))
                        f3.close()
                        best_acc = acc
                        torch.save(net, '../net_re_%03d.pth' % (epoch + 1))
                print('train_acc_list ', train_acc_list)
                print('val_acc_list ', val_acc_list)
                print('train_loss_list ', train_loss_list)
                print('val_loss_list ', val_loss_list)
            print("Training Finished, TotalEPOCH=%d" % epochs)
            f4 = open("../train_acc_list.txt", "w")
            f4.write(str(train_acc_list))
            f5 = open("../val_acc_list.txt", "w")
            f5.write(str(val_acc_list))
            f6 = open("../train_loss_list.txt", "w")
            f6.write(str(train_loss_list))
            f7 = open("../val_loss_list.txt", "w")
            f7.write(str(val_loss_list))
