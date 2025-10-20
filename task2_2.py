import torch
from torchsummary import summary
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义第一个模型
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.linear1 = nn.Linear(3, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# 定义第二个模型
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.output = nn.Linear(16 * 64 * 64, 2)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        temp = x.view(x.size()[0], -1)
        x = self.dp1(x)
        output = self.output(temp)
        return output, x

# 定义第三个模型
class MYVGG(nn.Module):
    def __init__(self, num_classes=2):
        super(MYVGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.output = nn.Linear(in_features=512 * 8 * 8, out_features=2)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.dp1(x)
        x = x.view(x.size(0), -1)
        output = self.output(x)
        return output, x

if __name__ == '__main__':

    # 实例化模型并打印摘要
    model1 = Model1().to(device)
    summary(model1, (3,))  # 输入形状为 (3,)，因为是全连接层

    model2 = CNN().to(device)
    summary(model2, (3, 256, 256))  # 输入形状为 (3, 256, 256)，因为是卷积网络

    model3 = MYVGG().to(device)
    summary(model3, (3, 224, 224))  # 输入形状为 (3, 224, 224)，因为是VGG网络