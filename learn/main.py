import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 100)
x_np = x.numpy()

y_relu = torch.relu(x).numpy()
y_sigmoid = torch.sigmoid(x).numpy()
y_tanh = torch.tanh(x).numpy()

plt.subplot(111)
plt.plot(x_np, y_relu)
