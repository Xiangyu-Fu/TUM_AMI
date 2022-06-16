# TODO: Open coco format files
from pycocotools.coco import COCO

import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random


# define the coco dataset process function
def coco_dataset_process(save_images=False, new_label_mode=False):
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

    # get the new annotation file
    if new_label_mode:
        for index, imgId in enumerate(imgIds):
            # get the image
            img = coco.loadImgs(imgId)[0]
            file_name = img['file_name']
            img_path = os.path.join(cocoRoot, f'Images/{file_name}')
            img_show = cv2.imread(img_path)

            plt.figure(1,  figsize=(12,8))

            # the left side of the image, show the image and the annotation
            plt.subplot(1, 2, 1)
            plt.text(0, -150, 'CLOSE this window and ENTER the category number,', fontsize=13)
            plt.text(0, -50, 'scratch = 1, dent = 2, rim = 3, other = 4', fontsize=13)
            plt.text(3000, 1800, f'you still have {len(imgIds) - index} images', fontsize=14)
            plt.axis('off')
            plt.imshow(img_show)
            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annIds)
            coco.showAnns(anns)

            # the right side of the image, show the detailed annotation
            pos = anns[0]['segmentation'][0]
            img_seg = img_show[pos[1]:pos[7], pos[0]:pos[2], :]

            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.imshow(img_seg)
            plt.show()

            # get new label to txt file
            label = int(input(f"Please input the label for {imgId}: "))
            with open(os.path.join("label_example.txt"), "a") as f:
                f.write(file_name + "," + str(label) + "\n")

            # save the annotation
            # ann = {'segmentation': [pos], 'area': 1, 'iscrowd': 0, 'image_id': img['id'], 'bbox': pos, 'category_id': 1, 'id': 1, 'ignore': 0, 'label': label}

    # show the information of the specific image
    img_index = 5
    img_to_show = imgIds[img_index]
    img_to_show_Info = coco.loadImgs(imgIds)[img_index]
    print(TAG, f'Image {img_to_show}\'s info: {img_to_show_Info}')

    # show the image
    imPath = os.path.join(cocoRoot, 'Images', img_to_show_Info['file_name'])
    im = cv2.imread(imPath)

    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(im)

    # get the damage area of the image
    annIds = coco.getAnnIds(imgIds=img_to_show_Info['id'])
    anns = coco.loadAnns(annIds)
    print(TAG, f'the annotation of the image: {anns}')

    coco.showAnns(anns)

    # show the image with the damage area
    pos = anns[0]['segmentation'][0]
    img_seg = im[pos[1]:pos[7], pos[0]:pos[2], :]

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(img_seg)
    plt.show()


if __name__ == '__main__':
    coco_dataset_process(save_images=False, new_label_mode=True)


