import cv2
import matplotlib.pyplot as plt
import os
path = "logo_stan.png"

im = cv2.imread(path)
im_seg = im[0:100, 0:100, :]

# save the image
cv2.imwrite(os.path.join("test/logo_stan_seg.png"), im_seg)


plt.imshow(im_seg)
plt.show()
