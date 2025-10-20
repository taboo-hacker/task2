import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from task2_1 import *
from task2_2 import *

# 训练网络
def train():
    train_loader, test_loader = load_data()
    epoch_num = 200

    # GPU计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MYVGG().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00004)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)

            optimizer.zero_grad()  # 梯度清零
            output = model(data)[0]  # 前向传播
            loss = criterion(output, target)  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model, '../tmp/cnn.pkl')


# 性能评估
def test():
    train_loader, test_loader = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('../tmp/cnn.pkl')  # load model
    total = 0
    current = 0

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)[0]

        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()

    print('Accuracy: %d %%' % (100 * current / total))


if __name__ == '__main__':
    train()
    test()
