from pickle import TRUE
from torchvision import datasets
import numpy as np
import torch
data_folder = './content/'
fmnist = datasets.FashionMNIST(data_folder, download=True, train=True)
tr_images = fmnist.data
tr_targers = fmnist.targets

import matplotlib.pyplot as plt
# plt.imshow(tr_images[0])
# plt.show()

from imgaug import augmenters as iaa
aug = iaa.Affine(scale=2, fit_output=True)

# plt.imshow(aug.augment_image(np.asarray(tr_images[0])))
# plt.title('Scaled image')
# plt.show()

# aug = iaa.Affine(translate_px=10, fit_output=True)
# plt.imshow(aug.augment_image(np.asarray(tr_images[0])))
# plt.title('TRANSLATED IMAGE BY 10 PIXELS')
# plt.show()

# aug = iaa.Affine(translate_px={'x':10, 'y':2}, fit_output=True)
# plt.imshow(aug.augment_image(np.asarray(tr_images[0])))
# plt.title('Translate of 10 pix, 2 over rows')
# plt.show()

# aug = iaa.Affine(rotate=30, fit_output=True, cval=255)
# plt.imshow(aug.augment_image(np.asarray(tr_images[0])))
# plt.title('Rotation of image by 30 degress')
# plt.show()

# plt.figure(figsize=(20,20))
# plt.subplot(151)
# aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
#                  mode='constant')

# plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
# plt.subplot(152)
# aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
#                  mode='constant')
# plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
# plt.subplot(153)
# aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
#                  mode='constant')
# plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
# plt.subplot(154)
# aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
#                  mode='constant')
# plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
# plt.show()

# aug = iaa.Multiply(0.5)
# plt.imshow(aug.augment_image(np.asanyarray(tr_images[0])), cmap='gray',vmin=0, vmax=255)
# plt.title('Pixels multiplied by 0.5')
# plt.show()

# aug = iaa.LinearContrast(0.5)
# plt.imshow(aug.augment_image(np.asanyarray(tr_images[0])), cmap='gray',vmin=0, vmax=255)
# plt.title('Pixels contrast by 0.5')
# plt.show()

# aug = iaa.GaussianBlur(sigma=1)
# plt.imshow(aug.augment_image(np.asanyarray(tr_images[0])), cmap='gray',vmin=0, vmax=255)
# plt.title('Gaussian blurring by 1')
# plt.show()

# plt.figure(figsize=(10,10))
# plt.subplot(121)
# aug = iaa.Dropout(p=0.2)
# plt.imshow(aug.augment_image(np.asanyarray(tr_images[0])), cmap='gray', \
#            vmin = 0, vmax = 255)
# plt.title('Random 20% pixel dropout')
# plt.subplot(122)
# aug = iaa.SaltAndPepper(0.2)
# plt.imshow(aug.augment_image(np.asanyarray(tr_images[0])), cmap='gray', \
#            vmin = 0, vmax = 255)
# plt.title('Random 20% salt and pepper noise')
# plt.show()

seq = iaa.Sequential([
      iaa.Dropout(p=0.2),
      iaa.Affine(rotate=(-30,30))
],random_order=True)

plt.imshow(seq.augment_image(np.asanyarray(tr_images[0])), cmap='gray', \
           vmin = 0, vmax = 255)
plt.title('Image augmented using a \nrandom order \
of the two augmentations')
plt.show()