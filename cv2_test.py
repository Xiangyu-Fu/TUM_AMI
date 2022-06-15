import cv2
import matplotlib.pyplot as plt
path = "logo_stan.png"

im = cv2.imread(path)
im_seg = im[0:100, 0:100, :]

plt.imshow(im_seg)
plt.show()
