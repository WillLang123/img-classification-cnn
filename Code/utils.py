import numpy as np
import os
import cv2
import copy
import constants as CONST
import pickle
import matplotlib.pyplot as plt

# plots training history (accuracy, loss)
def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)

    plt.plot(history['accuracy'])  # plot accuracy
    plt.plot(history['val_accuracy'])  # plot validation accuracy
    plt.title('model accuracy')  # title for accuracy graph
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])  # plot loss
    plt.plot(history['val_loss'])  # plot validation loss
    plt.title('model loss')  # title for loss graph
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

# processes the image (resize and normalize)
def process_image(directory, img_path):
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)  # read image
    image_copy = copy.deepcopy(image)  # copy image

    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))  # resize image
    image_std = image.astype('float') / 255.0  # normalize image
    return image_copy, image_std  # return processed image
