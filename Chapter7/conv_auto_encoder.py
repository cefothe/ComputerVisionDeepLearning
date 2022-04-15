from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
img_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                    transforms.Lambda(lambda x: x.to(device))
                                    ])

trn_ds = MNIST('./content/', transform=img_transform, \
               train=True, download=True)
val_ds = MNIST('./content/', transform=img_transform, \
               train=False, download=True)

batch_size = 128
trn_dl = DataLoader(trn_ds, batch_size=batch_size, \
                    shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, \
                    shuffle=False)

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
                            nn.Conv2d(1, 32, 3, stride=3, \
                                      padding=1), 
                            nn.ReLU(True),
                            nn.MaxPool2d(2, stride=2),
                            nn.Conv2d(32, 64, 3, stride=2, \
                                      padding=1), 
                            nn.ReLU(True),
                            nn.MaxPool2d(2, stride=1)
                        )
        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 3, \
                                           stride=2), 
                        nn.ReLU(True),
                        nn.ConvTranspose2d(32, 16, 5, \
                                         stride=3,padding=1), 
                        nn.ReLU(True),
                        nn.ConvTranspose2d(16, 1, 2, \
                                         stride=2,padding=1), 
                        nn.Tanh()
                    )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), \
                              lr=0.001, weight_decay=1e-5)

num_epochs = 5
log = Report(num_epochs)


def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss


for epoch in range(num_epochs):
    N = len(trn_dl)
    for ix, (data, _) in enumerate(trn_dl):
        loss = train_batch(data, model, criterion, optimizer)
        log.record(pos=(epoch + (ix+1)/N), \
                   trn_loss=loss, end='\r')
        
    N = len(val_dl)
    for ix, (data, _) in enumerate(val_dl):
        loss = validate_batch(data, model, criterion)
        log.record(pos=(epoch + (ix+1)/N), \
                   val_loss=loss, end='\r')
        
    log.report_avgs(epoch+1)

log.plot_epochs(log=True)
for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    _im = model(im[None])[0]
    fig, ax = plt.subplots(1, 2, figsize=(3,3))
    show(im[0], ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title='prediction')
    plt.tight_layout()
    plt.show()

latent_vectors = []
classes = []
for im,clss in val_dl:
    latent_vectors.append(model.encoder(im).view(len(im),-1))
    classes.extend(clss)
latent_vectors = torch.cat(latent_vectors).cpu()\
                      .detach().numpy()

from sklearn.manifold import TSNE
tsne = TSNE(2)
clustered = tsne.fit_transform(latent_vectors)
fig = plt.figure(figsize=(12,10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(*zip(*clustered), c=classes, cmap=cmap)
plt.colorbar(drawedges=True)
plt.show()