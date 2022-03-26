import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7], [11], [15]]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

model = nn.Sequential(
    nn.Linear(2,8),
    nn.ReLU(),
    nn.Linear(8,1)
).to(device)
from torchsummary import summary

summary(model, torch.zeros(1,2))

loss_func = nn.MSELoss()
from torch.optim import SGD
opt = SGD(model.parameters(), lr = 0.001)

import time
start = time.time()
loss_history = []
for _ in range(50):
    for ix,iy in dl:
        opt.zero_grad()
        loss_value = loss_func(model(ix),iy)
        loss_value.backward()
        opt.step()
        loss_history.append(loss_value)
end = time.time()

print(end - start)

val = [[8,9],[10,11],[1.5,2.5]]

# model(torch.tensor(val).float().to(device))

print(model(torch.tensor(val).float().to(device)))

torch.save(model.state_dict(), 'mymodel.pth')