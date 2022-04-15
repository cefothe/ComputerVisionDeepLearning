import torch
from torch import nn
from torch.utils.data import TensorDataset,Dataset,DataLoader
from torch.optim import SGD, Adam
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torchvision import datasets
import numpy as np, cv2
import matplotlib.pyplot as plt

from glob import glob
from imgaug import augmenters as iaa

tfm = iaa.Sequential(iaa.Resize(28))

class XO(Dataset):
    def __init__(self, folder):
        self.files = glob(folder)
    def __len__(self): return len(self.files)
    def __getitem__(self, ix):
        f = self.files[ix]
        im = tfm.augment_image(cv2.imread(f)[:,:,0])
        im = im[None]
        cl = f.split('/')[-1].split('@')[0] == 'x'
        return torch.tensor(1 - im/255).to(device).float(), \
                       torch.tensor([cl]).float().to(device)
# data = XO('./all/*')
# R, C = 7,7
# fig, ax = plt.subplots(R, C, figsize=(5,5))
# for label_class, plot_row in enumerate(ax):
#     for plot_cell in plot_row:
#         plot_cell.grid(False); plot_cell.axis('off')
#         ix = np.random.choice(1000)
#         im, label = data[ix]
#         print()
#         plot_cell.imshow(im[0].cpu(), cmap='gray')
# plt.tight_layout()
# plt.show()

from torch.optim import SGD, Adam
def get_model():
    model = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3200, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ).to(device)

    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer

from torchsummary import summary
model, loss_fn, optimizer = get_model()
summary(model, torch.zeros(1,1,28,28))

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    opt.step()
    opt.zero_grad()
    return batch_loss.item(), is_correct[0]

trn_dl = DataLoader(XO('./all/*'), batch_size=32,drop_last=True)

model, loss_fn, optimizer = get_model()
for epoch in range(5):
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, \
                                        loss_fn)

im, c = trn_dl.dataset[2]
plt.imshow(im[0].cpu())
plt.show()

first_layer = nn.Sequential(*list(model.children())[:1])
intermediate_output = first_layer(im[None])[0].detach()
fig, ax = plt.subplots(8, 8, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.set_title('Filter: '+str(ix))
    axis.imshow(intermediate_output[ix].cpu())
plt.tight_layout()
plt.show()

x, y = next(iter(trn_dl))
x2 = x[y==0]
x2 = x2.view(-1,1,28,28)
first_layer = nn.Sequential(*list(model.children())[:1])
first_layer_output = first_layer(x2).detach()
n = 3
fig, ax = plt.subplots(n, n, figsize=(10,10))
for ix, axis in enumerate(ax.flat):
    axis.imshow(first_layer_output[ix,4,:,:].cpu())
    axis.set_title(str(ix))
plt.tight_layout()
plt.show()