import numpy as np
import os
from random import shuffle
import constants as CONST
import cv2
from PIL import Image

# gets size statistics of images in directory
def get_size_statistics(dir):
    heights = []
    widths = []
    img_count = 0
    DIR = dir
    for img in os.listdir(dir):
        path = os.path.join(DIR, img)
        data = cv2.imread(path)
        data = np.array(Image.open(path))
        heights.append(data.shape[0])  # height of image
        widths.append(data.shape[1])  # width of image
        img_count += 1
    avg_height = sum(heights) / len(heights)  # average height
    avg_width = sum(widths) / len(widths)  # average width
    print("Average Height: " + str(avg_height))
    print("Max Height: " + str(max(heights)))  # max height
    print("Min Height: " + str(min(heights)))  # min height
    print('\n')
    print("Average Width: " + str(avg_width))
    print("Max Width: " + str(max(widths)))  # max width
    print("Min Width: " + str(min(widths)))  # min width

get_size_statistics(CONST.TRAIN_DIR_1)  # get stats for train1
get_size_statistics(CONST.TRAIN_DIR_2)  # get stats for train2

# labels image based on filename
def label_img(name):
    word_label = name.split('.')[0]
    label = CONST.LABEL_MAP[word_label]
    label_arr = np.zeros(2)
    label_arr[label] = 1
    return label_arr

# loads and prepares image data
def prep_and_load_data(dir):
    print(os.getcwd())  # for debugging
    DIR = dir
    data = []
    image_paths = os.listdir(DIR)
    shuffle(image_paths)  # shuffle image paths for randomness
    count = 0
    for img_path in image_paths:
        label = label_img(img_path)  # get label for image
        path = os.path.join(DIR, img_path)
        image = cv2.imread(path)  # read image
        image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))  # resize image
        image = image.astype('float') / 255.0  # normalize image
        data.append([image, label])  # add image and label to data
        # print(count, image.shape, label.shape,data[count][1])  # print shape for debugging
        count += 1
        if count == CONST.DATA_SIZE:  # limit to DATA_SIZE images
            break
    shuffle(data)  # shuffle data after loading
    data = np.array(data, dtype=object)
    return data

if __name__ == "__main__":
    prep_and_load_data(CONST.TRAIN_DIR_1)  # load and prep data for train1
    prep_and_load_data(CONST.TRAIN_DIR_2)  # load and prep data for train2
