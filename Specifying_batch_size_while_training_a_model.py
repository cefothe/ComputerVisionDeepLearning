from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

x =[[1,2], [3,4], [5,6], [7,8]]
y = [[3],[7],[11],[15]]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for par in mynet.parameters():
#     print(par)


x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7], [11], [15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

X = X.to(device)
Y = Y.to(device)

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

ds = MyDataset(X,Y)
dl = DataLoader(ds, batch_size=2, shuffle=True)

for x,y in dl:
    print(x,y)

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

mynet = MyNeuralNet().to(device)
loss_func = nn.MSELoss()
from torch.optim import SGD
opt = SGD(mynet.parameters(), lr = 0.001)

import time
start = time.time()
loss_history = []
for _ in range(50):
    for data in dl:
        x, y = data
        opt.zero_grad()
        loss_value = loss_func(mynet(x), y)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value.detach().numpy())
end = time.time()

print(end - start)

val_x = [[10,11]]
val_x = torch.tensor(val_x).float().to(device)

print(mynet(val_x))