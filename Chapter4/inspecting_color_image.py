import cv2, matplotlib.pylab as plt

img = cv2.imread('Hemanvi.jpeg')
img = img[50:250, 40:240,:]

img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray_small = cv2.resize(img, (25,25))
plt.imshow(img)
plt.show()