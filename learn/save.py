import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.4)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1, 'net.pkl')  # 保存整个网络
    torch.save(net1.state_dict(), 'net_params.pkl')  # 只保存网络中的参数 (速度快, 占内存少)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.scatter(x.detach().numpy(), y.detach().numpy())
    plt.plot(x.detach().numpy(), prediction.detach().numpy(), 'r-')


def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    plt.subplot(1, 3, 2)
    plt.scatter(x.detach().numpy(), y.detach().numpy())
    plt.plot(x.detach().numpy(), prediction.detach().numpy(), 'r-')


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)
    plt.subplot(1, 3, 3)
    plt.scatter(x.detach().numpy(), y.detach().numpy())
    plt.plot(x.detach().numpy(), prediction.detach().numpy(), 'r-')
    plt.show()


save()
restore_net()
restore_params()
