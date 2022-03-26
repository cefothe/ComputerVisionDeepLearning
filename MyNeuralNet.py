import torch.nn as nn
import torch
from torch.optim import SGD
import matplotlib.pyplot as plt


class MyNeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)

    def forward(self, x):
        # x= x @ self.input_to_hidden_layer
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mynet = MyNeuralNet().to(device)

# for par in mynet.parameters():
#     print(par)


x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7], [11], [15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

X = X.to(device)
Y = Y.to(device)

loss_func = nn.MSELoss()

opt = SGD(mynet.parameters(), lr = 0.001)

import time
start = time.time()
loss_history = []
for _ in range(50):
    opt.zero_grad()
    loss_value = loss_func(mynet(X), Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value.detach().numpy())

end = time.time()

print(end - start)

# plt.plot(loss_history)
# plt.title("Loss variation over increasing epochs")
# plt.xlabel("epochs")
# plt.ylabel("loss value")
# plt.show()