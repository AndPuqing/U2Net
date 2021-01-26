import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # activity function
import numpy as np

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 小问题，why 加维
y = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data shape=(100,1)

# 画图
plt.scatter(x.numpy(), y.numpy())
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):  # 正向传播
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.predict(x)  # 输出值
        return x


net = Net(1, 10, 1)
print(net)
'''
Net(
  (hidden): Linear(in_features=1, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=1, bias=True)
)
'''
plt.ion()
plt.show()

# optimizer 是训练工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.4)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)
lossnu = np.array([1])  # 记录loss值

for t in range(1000):
    prediction = net(x)  # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)  # 计算两者的误差，注意顺序

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    lossnu = np.append(lossnu, loss.detach().numpy())  # 添加loss值，注意转换

    # 画图
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.numpy(), y.numpy())
        plt.plot(x.detach().numpy(), prediction.detach().numpy(), 'r-')
        plt.pause(0.1)
        print(loss.detach().numpy())

plt.close()
plt.plot(np.arange(np.size(lossnu)), lossnu)
plt.ioff()
plt.show()
