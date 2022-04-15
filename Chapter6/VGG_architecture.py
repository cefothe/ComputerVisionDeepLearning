import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms,models,datasets
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = models.vgg16(pretrained=True).to(device)
summary(model, torch.zeros(1,3,224,224))