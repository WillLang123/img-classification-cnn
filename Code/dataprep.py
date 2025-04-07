import numpy as np
import os
from random import shuffle
import constants as CONST
import cv2
import copy

def processImage(dir, imagePath):
    path = os.path.join(dir, imagePath)
    normImage = cv2.imread(path)  # read image
    imageCopy = copy.deepcopy(normImage)  # copy image
    normImage = cv2.resize(normImage, (CONST.IMAGESIZE, CONST.IMAGESIZE))  # resize image
    normImage = normImage.astype('float') / 255.0  # normalizes image to somewhere between 0 and 1
    return imageCopy, normImage  # return processed image and copy of original

# labels image based on filename
def labelImage(name):
    imageLabel = name.split('.')[0]
    label = CONST.LABELMAP[imageLabel]
    labels = np.zeros(2)
    labels[label] = 1
    return labels

# loads and prepares image data
def prepData(dir):
    print(os.getcwd())  # for debugging
    DIR = dir
    data = []
    imagePaths = os.listdir(DIR)
    shuffle(imagePaths)  # shuffle image paths for randomness
    count = 0
    for imagePath in imagePaths:
        label = labelImage(imagePath)  # get label for image
        path = os.path.join(DIR, imagePath)
        image = cv2.imread(path)  # read image
        image = cv2.resize(image, (CONST.IMAGESIZE, CONST.IMAGESIZE))  # resize image
        image = image.astype('float') / 255.0  # normalize image
        data.append([image, label])  # add image and label to data
        # print(count, image.shape, label.shape,data[count][1])  # print shape for debugging
        count += 1
        if count == CONST.DATASIZE:  # limit to DATASIZE images
            break
    shuffle(data)  # shuffle data after loading
    data = np.array(data, dtype=object) #need it to be object datatype for some numpy reason
    return data
