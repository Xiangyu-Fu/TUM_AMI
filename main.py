# TODO: Open coco format files
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random


# define the coco dataset process function
def coco_dataset_process(save_images=False, save_annotations=False):
    """
    This function processes the coco dataset and saves the images and annotations.
    :param save_images: default False. If true, the images will be saved.
    :param save_annotations: default False. If true, the annotations will be saved.
    :return: nothing.
    """
    TAG = "(coco_processing)"
    # Define the path to the dataset
    cocoRoot = "E:/AMI Project/Data/"

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

    # save the annotated images
    if save_images:
        for imgId in imgIds:
            # save the image
            img = coco.loadImgs(imgId)[0]
            file_name = img['file_name']
            img_path = os.path.join(cocoRoot, f'Images/{file_name}')
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(cocoRoot, f"Images/annotated/{img_name}"), img)
    # TODO: save all the new annotations

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
    img_seg = im[pos[1]:pos[7], pos[0]:pos[2], :]

    plt.imshow(img_seg)
    plt.show()


if __name__ == '__main__':
    coco_dataset_process(save_images=True)


