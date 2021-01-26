import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # activity function
import numpy as np

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

# 画图
plt.scatter(x.detach().numpy()[:, 0], x.detach().numpy()[:, 1], c=y.detach().numpy(), s=100, lw=0)
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


net = Net(2, 11, 2)
print(net)
'''
Net(
  (hidden): Linear(in_features=2, out_features=10, bias=True)
  (predict): Linear(in_features=10, out_features=2, bias=True)
)
'''
plt.ion()
plt.show()

# optimizer 是训练工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
loss_func = torch.nn.CrossEntropyLoss()  # 预测值和真实值的误差计算公式 (均方差)
lossnu = np.array([1])  # 记录loss值

for t in range(150):
    out = net(x)  # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(out, y)  # 计算两者的误差，注意顺序

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
    lossnu = np.append(lossnu, loss.detach().numpy())  # 添加loss值，注意转换

    # 画图
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out, dim=1), 1)[1]
        pred_y = prediction.numpy().squeeze()
        target_y = y.numpy()
        plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=100, lw=0)
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
        print(loss)

plt.close()
plt.plot(np.arange(np.size(lossnu)), lossnu)
plt.ioff()
plt.show()
