import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F  # activity function
import numpy as np

# 假数据
n_data = torch.ones(100, 2)  # 数据的形态，100组，每组两个量
x0 = torch.normal(2 * n_data, 1)  # 生成第一种类型的数据
y0 = torch.zeros(100)  # 其标志值
x1 = torch.normal(-2 * n_data, 1)  # 第二种类型值
y1 = torch.ones(100)  # 标志值
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 对应x,y
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # 标志值合并

# 画图
plt.scatter(x.detach().numpy()[:, 0], x.detach().numpy()[:, 1], c=y.detach().numpy(), s=100, lw=0)
plt.show()

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print(net)
'''
Sequential(
  (0): Linear(in_features=2, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=2, bias=True)
)
'''
plt.ion()
plt.show()

# optimizer 是训练工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 传入 net 的所有参数, 学习率
# 这里算误差不是one-hot形式，而是1D Tensor，(batch，)
# 但是预测值是2D tensor的
loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
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
        '''
        这里看出softmax是将输出值映射到了和为1的空间，dim规定对哪一维操作，max输出有两个tensor，见文档
        '''
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
