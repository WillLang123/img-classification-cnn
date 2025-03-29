import numpy as np
import os
import cv2
import copy
import constants as CONST
import pickle
import matplotlib.pyplot as plt

# processes the image (resize and normalize)
def process_image(directory, imagePath):
    path = os.path.join(directory, imagePath)
    image = cv2.imread(path)  # read image
    imageCopy = copy.deepcopy(image)  # copy image

    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))  # resize image
    normImage = image.astype('float') / 255.0  # normalize image
    return imageCopy, normImage  # return processed image
