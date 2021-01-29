import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库模块以及图形操作库
from tqdm import tqdm  # 进度条
import matplotlib.pyplot as plt

EPOCH = 1  # 训练轮数
BATCH_SIZE = 50  # 一个Batch大小
LR = 0.001  # 学习率
DOWNLOAD_MNIST = False  # 设置是否下载数据 True:下载

# Mnist 手写数字
# torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)
"""
Args:
    root (string): Root directory of dataset where ``MNIST/processed/training.pt``
        and  ``MNIST/processed/test.pt`` exist.
    train (bool, optional): If True, creates dataset from ``training.pt``,
        otherwise from ``test.pt``.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    transforms.ToTensor(),转换 PIL.Image or numpy.ndarray 成torch.FloatTensor (C x H x W), 
    训练的时候 normalize 成 [0.0, 1.0] 区间
"""
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

# plot one example
# print(train_data.data.size())  # (60000, 28, 28)
# print(train_data.targets.size())  # (60000)
# plt.imshow(train_data.data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.targets[0])
# plt.show()

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# load test data
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.  # 由于没有加入通过转换，这里手动转换
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.con1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = x.view(x.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)
'''
(con1): Sequential(
    (0): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (con2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (out): Linear(in_features=1568, out_features=10, bias=True)
)
'''
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

for epoch in tqdm(range(EPOCH)):  # 这里利用tqdm记录时间
    for step, (x, y) in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
        output = cnn(x)  # cnn output
        loss = loss_func(output, y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # 每个Batch进行测试
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].detach().numpy()  # output(2000,10)->(2000),max会对dim维度比较大小，返回数值，和标签
            accuracy = float((pred_y == test_y.detach().numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch:', epoch, '|train loss:%.4f' % loss.detach().numpy(), '|test accuracy:%.2f' % accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].detach().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
