from torchvision import datasets
import torch
import numpy as np
data_folder = '~/content/' 
fmnist = datasets.FashionMNIST(data_folder, download=True, \
                                                train=True)
tr_images = fmnist.data
tr_targets = fmnist.targets

from imgaug import augmenters as iaa
aug = iaa.Sequential([
    iaa.Affine(translate_px={'x':(-10,10)},mode='constant')
])

for i in range(32):
  aug.augment_image(np.asarray(tr_images[i]))

x = aug.augment_images(tr_images[:32].numpy())
# device = "cuda" if torch.cuda.is_available() else 'cpu'

# from torch.utils.data import Dataset, DataLoader
# class FMNISTDataset(Dataset):
#     def __init__(self, x, y, aug=None):
#         self.x, self.y = x, y
#         self.aug = aug

#     def __getitem__(self, ix):
#         x, y = self.x[ix], self.y[ix]
#         return x, y

#     def __len__(self): return len(self.x)

#     def collate_fn(self, batch):
#         ims, classes = list(zip(*batch))
#         if self.aug : ims=self.aug.augment_images(images=ims)
#         ims = torch.tensor(ims)[:,None,:,:].to(device)/255.
#         classes = torch.tensor(classes).to(device)
#         return ims, classes        

# train = FMNISTDataset(tr_images, tr_targets, aug)

# trn_dl = DataLoader(train, batch_size=64 ,collate_fn=train.collate_fn, shuffle = True)
