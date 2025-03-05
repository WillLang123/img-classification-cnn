import numpy as np
from data_prep import prep_and_load_data
from model import get_model
import constants as CONST
import pickle
import os
import cv2
from tensorflow.keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import copy

def plotter(history_file):
    with open(history_file, 'rb') as file:
        history = pickle.load(file)
    
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.savefig('1000_15epoch_accuracy.png')

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig('1000_15epoch_loss.png')


def video_write(model):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("./prediction.mp4", fourcc, 1.0, (400,400))
    val_map = {1: 'Dog', 0: 'Cat'}

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20,20)
    fontScale = 0.5
    fontColor = (255,255,255)
    lineType  = 2

    DIR = CONST.TEST_DIR
    image_paths = os.listdir(DIR)
    image_paths = image_paths[:200]
    count = 0
    for img_path in image_paths:
        image, image_std = process_image(DIR, img_path)
        
        image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
        pred = model.predict([image_std])
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        s = val_map[arg_max[0]] + ' - ' + str(max_val[0]*100) + '%'
        cv2.putText(image, s, 
            location, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        image = cv2.resize(image, (400, 400))
        out.write(image)
        
        count += 1
        print(count)
    out.release()



def process_image(directory, img_path):
    path = os.path.join(directory, img_path)
    image = cv2.imread(path)
    image_copy = copy.deepcopy(image)
    
    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))
    image_std = image.astype('float') / 255.0
    return image_copy, image_std


if __name__ == "__main__":
    data1 = prep_and_load_data(CONST.TRAIN_DIR_1)
    data2 = prep_and_load_data(CONST.TRAIN_DIR_2)

    train_size = int(CONST.DATA_SIZE * CONST.SPLIT_RATIO)
    print('data1', len(data1), train_size)
    print('data2', len(data2), train_size)

    train_data1 = data1[:train_size]
    train_images1 = np.array([i[0] for i in train_data1]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    train_labels1 = np.array([i[1] for i in train_data1])
    print('train data1 fetched..')

    train_data2 = data2[:train_size]
    train_images2 = np.array([i[0] for i in train_data2]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    train_labels2 = np.array([i[1] for i in train_data2])
    print('train data2 fetched..')

    test_data1 = data1[train_size:]
    test_images1 = np.array([i[0] for i in test_data1]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    test_labels1 = np.array([i[1] for i in test_data1])
    print('data1 fetched..')

    test_data2 = data2[train_size:]
    test_images2 = np.array([i[0] for i in test_data2]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)
    test_labels2 = np.array([i[1] for i in test_data2])
    print('data2 fetched..')

    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

    model1 = get_model()
    print('dataset 1 training started...')
    history = model1.fit(train_images1, train_labels1, batch_size = 50, epochs = 15, verbose = 1, validation_data=(test_images1, test_labels1), callbacks=[tensorboard_callback])
    print('dataset 1 training done...')

    model2 = get_model()
    print('dataset 2 training started...')
    history = model2.fit(train_images2, train_labels2, batch_size = 50, epochs = 15, verbose = 1, validation_data=(test_images2, test_labels2), callbacks=[tensorboard_callback])
    print('dataset 2 training done...')

    model1.save('1000_1.h5')
    model2.save('1000_2.h5')

    history_file = '1000_history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    plotter(history_file)
    video_write(model1)
    video_write(model2)




