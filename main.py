# TODO: Open coco format files
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random

TAG = "(main)"
# Define the path to the dataset
cocoRoot = "E:/AMI Project/Data"

# Load COCO dataset
annFile = os.path.join(cocoRoot, f'Annotations/annotated_functional_test3_fixed.json')
print(TAG, f'Annotation file: {annFile}')

# Load annotations
coco = COCO(annFile)

# load the ID of the category
ids = coco.getCatIds('damage')

# Load image IDs
imgIds = coco.getImgIds(catIds=ids)
print(TAG, f'Number of images: {len(imgIds)}')

# get the information of the first image
imgId = imgIds[0]
imgInfo = coco.loadImgs(imgIds)[0]
print(TAG, f'Image {imgId}\'s info: {imgInfo}')

# show the image
imPath = os.path.join(cocoRoot, 'Images', imgInfo['file_name'])
im = cv2.imread(imPath)
plt.imshow(im)


# get the damage area of the image
annIds = coco.getAnnIds(imgIds=imgInfo['id'])
anns = coco.loadAnns(annIds)
print(TAG, f'the annotation of the image: {anns}')

coco.showAnns(anns)
plt.show()


# show the image with the damage area
pos = anns[0]['segmentation'][0]
print(pos[1], pos[7])
img_seg = im[ pos[1]:pos[7], pos[0]:pos[2], :]

plt.imshow(img_seg)
plt.show()

