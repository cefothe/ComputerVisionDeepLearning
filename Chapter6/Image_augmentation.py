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

aug = iaa.Affine(rotate=30, fit_output=True, cval=255)
plt.imshow(aug.augment_image(np.asarray(tr_images[0])))
plt.title('Rotation of image by 30 degress')
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(151)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')

plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
plt.subplot(152)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')
plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
plt.subplot(153)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')
plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
plt.subplot(154)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \
                 mode='constant')
plt.imshow(aug.augment_image(np.asarray(tr_images[0])), cmap='gray')
plt.show()