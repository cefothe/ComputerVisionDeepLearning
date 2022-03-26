import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = nn.Sequential(
    nn.Linear(2,8),
    nn.ReLU(),
    nn.Linear(8,1)
).to(device)

state_dict = torch.load('mymodel.pth')
model.load_state_dict(state_dict)
model.to(device)

val = [[8,9],[10,11],[1.5,2.5]]

print(model(torch.tensor(val).float().to(device)))