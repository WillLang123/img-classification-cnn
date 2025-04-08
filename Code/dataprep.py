import numpy as np
import os
from random import shuffle
import constants as CONST
from cv2 import imread, resize
from copy import deepcopy

def processImage(dir, imagePath):
    imageDim = (CONST.IMAGESIZE, CONST.IMAGESIZE)
    path = os.path.join(dir,imagePath)
    normImage = imread(path)  # reads the image
    imageCopy = deepcopy(normImage)  # copy image
    normImage = resize(normImage, imageDim)  # resize image
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
    data = []
    imagePaths = os.listdir(dir)
    shuffle(imagePaths)  # shuffle image paths for randomness
    count = 0
    outputDim = (CONST.IMAGESIZE, CONST.IMAGESIZE)
    for imagePath in imagePaths:
        label = labelImage(imagePath)  # get label for image
        path = os.path.join(dir, imagePath)
        image = imread(path)  # read image
        image = resize(image, outputDim)  # resize image
        normImage = image.astype('float') / 255.0  # normalize image
        data.append([normImage, label])  # add image and label to data
        # print(count, image.shape, label.shape,data[count][1])  # print shape for debugging
        count += 1
        if count == CONST.DATASIZE:  # limit to DATASIZE images
            break
    shuffle(data)  # shuffle data after loading
    data = np.array(data, dtype=object) #need it to be object datatype for some numpy reason
    return data
