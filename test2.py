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
import warnings
warnings.filterwarnings("ignore")
# 定义读取文件的格式


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')  # 删除本行string 字符串末尾的指定字符
            words = line.split()  # 用split将该行分割成列表
            imgs.append((words[0], int(words[1])))
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
test_transforms = transforms.Compose([transforms.Resize((112, 112)),
                                      transforms.ToTensor(),
                                      ])


def make_model():
    resmodel = torch.load(
        './net.pth', map_location={'cuda0:': 'cuda:0'})
#     net = resmodel.cuda(device=device_ids[2])#将模型从CPU发送到GPU,如果没有GPU则删除该行
    return resmodel

# 读取ImageFolder图像


def read_png(ImageFolder_data):
    label = []
    pid_index = []
    for sub_file in ImageFolder_data.imgs:
        pid_index.append(sub_file[0].split('/')[-1])
        label.append(sub_file[1])
    return pid_index, label


#### 读取ImageFolder的feature and result
def read_ImageFolder(ImageFolder_data, num):
    i = 0
    feature_all = np.zeros([num, 512])
    model = make_model()
    result_all = []
    score_all = []
    for sub_file in ImageFolder_data:
        image = sub_file[0][0]
        tensor = image.resize_(1, 3, 112, 112)
        tensor = tensor.cuda(device=device_ids[0])  # 将数据发送到GPU，数据和模型在同一个设备上运行

        result, score, feature = extract_feature(model, tensor)
        result_all.append(result)
        score_all.append(score)
        i = i+1
        feature_all[i-1, :] = feature

    return result_all, score_all, feature_all


def extract_feature(resmodel, tensor):
    resmodel.fc = torch.nn.LeakyReLU(0.1)
    resmodel.eval()

    result = resmodel(Variable(tensor))
    result_npy = result.data.cpu().numpy()
    max_index = np.argmax(result_npy[0])

    score = [result_npy[0][i] for i in range(2)]   # 2表示类别数量

    x = resmodel.module.conv1(Variable(tensor))
    x = resmodel.module.bn1(x)
    x = resmodel.module.relu(x)
    x = resmodel.module.maxpool(x)
    x = resmodel.module.layer1(x)
    x = resmodel.module.layer2(x)
    x = resmodel.module.layer3(x)
    x = resmodel.module.layer4(x)
    x = resmodel.module.avgpool(x)
    x = np.array(x.data.cpu())
    x = np.reshape(x, (1, 512))

    return max_index, score, x


# 数据集加载方式设置
dir = './train_test_data/'
train_data = MyDataset(txt=dir+'train.txt', transform=test_transforms)
val_data = MyDataset(txt=dir+'val.txt', transform=test_transforms)
test_data = MyDataset(txt=dir+'test.txt', transform=test_transforms)

train_loader = DataLoader(dataset=train_data)
val_loader = DataLoader(dataset=val_data)
test_loader = DataLoader(dataset=test_data)

device_ids = [0]
a = make_model()

# 训练集预测结果
num_png = len(open(dir+'train.txt', 'rU').readlines())
pid_index, label = read_png(train_data)
result_all, score_all, feature_all = read_ImageFolder(
    train_data, num_png)

df_result = pd.DataFrame(result_all, columns=["pre"], index=pid_index)   # 预测标签
df_pid = pd.DataFrame(pid_index, columns=["pid"], index=pid_index)
df_label = pd.DataFrame(label, columns=["label"], index=pid_index)  # 真实标签
score_all = np.array(score_all)
df_score = pd.DataFrame(score_all, columns=["0", '1'], index=pid_index)

df = pd.concat([df_label, df_result, df_score], axis=1)
sum = np.exp(df['0']) + np.exp(df['1'])
df['0'] = np.exp(df['0'])/sum
df['1'] = np.exp(df['1'])/sum
# df.columns = ['path', 'label', 'pre', 'score0', 'score']
df.to_csv('./train_result.csv')


# 验证集预测结果
num_png = len(open(dir+'val.txt', 'rU').readlines())
pid_index, label = read_png(val_data)
result_all, score_all, feature_all = read_ImageFolder(
    val_data, num_png)

df_result = pd.DataFrame(result_all, columns=["pre"], index=pid_index)   # 预测标签
df_pid = pd.DataFrame(pid_index, columns=["pid"], index=pid_index)
df_label = pd.DataFrame(label, columns=["label"], index=pid_index)  # 真实标签
score_all = np.array(score_all)
df_score = pd.DataFrame(score_all, columns=["0", '1'], index=pid_index)

df = pd.concat([df_label, df_result, df_score], axis=1)
sum = np.exp(df['0']) + np.exp(df['1'])
df['0'] = np.exp(df['0'])/sum
df['1'] = np.exp(df['1'])/sum
# df.columns = ['path', 'label', 'pre', 'score0', 'score']
df.to_csv('./val_result.csv')


# 测试集预测结果
num_png = len(open(dir+'test.txt', 'rU').readlines())
pid_index, label = read_png(test_data)
result_all, score_all, feature_all = read_ImageFolder(
    test_data, num_png)

df_result = pd.DataFrame(result_all, columns=["pre"], index=pid_index)   # 预测标签
df_pid = pd.DataFrame(pid_index, columns=["pid"], index=pid_index)
df_label = pd.DataFrame(label, columns=["label"], index=pid_index)  # 真实标签
score_all = np.array(score_all)
df_score = pd.DataFrame(score_all, columns=["0", '1'], index=pid_index)

df = pd.concat([df_label, df_result, df_score], axis=1)
sum = np.exp(df['0']) + np.exp(df['1'])
df['0'] = np.exp(df['0'])/sum
df['1'] = np.exp(df['1'])/sum
# df.columns = ['path', 'label', 'pre', 'score0', 'score']
df.to_csv('./test_result.csv')
