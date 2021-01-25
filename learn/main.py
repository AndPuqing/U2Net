import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
plt.scatter(x.numpy(), y.numpy())
plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
print(net)
plt.ion()
plt.show()
optimizer = torch.optim.SGD(net.parameters(), lr=0.4)
loss_func = torch.nn.MSELoss()
lossnu = np.array([1])
for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lossnu = np.append(lossnu, loss.detach().numpy())
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
