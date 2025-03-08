import numpy as np
import pickle
import os
import cv2
from tensorflow.keras.callbacks import TensorBoard
import constants as CONST
from data_prep import prep_and_load_data
from model import get_model
from svm import svm_predict, load_svm_model, svm_train
from utils import plotter, process_image  # Import functions directly from utils

# writes predictions to video
def video_write(model, i):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # codec
    filename = "./prediction"+str(i)+".mp4"
    out = cv2.VideoWriter(filename, fourcc, 1.0, (400, 400))  # output video

    val_map = {1: 'Dog', 0: 'Cat'}  # mapping for predictions

    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (20, 20)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    DIR = CONST.TEST_DIR
    image_paths = os.listdir(DIR)
    image_paths = image_paths[:100]  # limit to 200 images
    count = 0
    for img_path in image_paths:
        image, image_std = process_image(DIR, img_path)  # process image
        
        image_std = image_std.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # reshape image
        pred = model.predict([image_std])  # make prediction
        arg_max = np.argmax(pred, axis=1)
        max_val = np.max(pred, axis=1)
        s = val_map[arg_max[0]] + ' - ' + str(max_val[0] * 100) + '%'
        cv2.putText(image, s, location, font, fontScale, fontColor, lineType)  # add text to image
        
        image = cv2.resize(image, (400, 400))  # resize image
        out.write(image)  # write to video
        
        count += 1
        print(count)  # print progress
    out.release()  # release video

# loads and preps images for models
if __name__ == "__main__":

    # loads and preps images for models
    data1 = prep_and_load_data(CONST.TRAIN_DIR_1)  # loading images from train1
    # data2 = prep_and_load_data(CONST.TRAIN_DIR_2)  # loading images from train2

    # figures out data size and what goes where
    train_size = int(CONST.DATA_SIZE * CONST.SPLIT_RATIO)
    print('data1', len(data1), train_size)  # size of data1
    # print('data2', len(data2), train_size)  # size of data2

    # sets up tensorboard callback to use for CNN model
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

    # splits the data into training and testing sets for both datasets
    train_data1 = data1[:train_size]  # training data from dataset 1
    train_images1 = np.array([i[0] for i in train_data1]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize images
    train_labels1 = np.array([i[1] for i in train_data1])  # labels for dataset 1

    # train_data2 = data2[:train_size]  # training data from dataset 2
    # train_images2 = np.array([i[0] for i in train_data2]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize images
    # train_labels2 = np.array([i[1] for i in train_data2])  # labels for dataset 2

    # splits the data into test sets for both datasets
    test_data1 = data1[train_size:]  # test data from dataset 1
    test_images1 = np.array([i[0] for i in test_data1]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize test images
    test_labels1 = np.array([i[1] for i in test_data1])  # test labels for dataset 1

    # test_data2 = data2[train_size:]  # test data from dataset 2
    # test_images2 = np.array([i[0] for i in test_data2]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize test images
    # test_labels2 = np.array([i[1] for i in test_data2])  # test labels for dataset 2

    # gets model to use
    model1 = get_model()  # get CNN model for dataset 1
    print('dataset 1 training started...')
    history = model1.fit(
        train_images1, train_labels1, batch_size=50, epochs=15, verbose=1,
        validation_data=(test_images1, test_labels1), callbacks=[tensorboard_callback]
    )
    print('dataset 1 training done...')

    # trains model
    history_file = '1000_history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)  # save history of training

    # plots training history
    plotter(history_file)  # plot accuracy and loss graphs

    # # trains model 2
    # model2 = get_model()  # get CNN model for dataset 2
    # print('dataset 2 training started...')
    # history = model2.fit(
    #     train_images2, train_labels2, batch_size=50, epochs=15, verbose=1,
    #     validation_data=(test_images2, test_labels2), callbacks=[tensorboard_callback]
    # )
    # print('dataset 2 training done...')

    # # trains model
    # history_file = '1000_history.pickle'
    # with open(history_file, 'wb') as file:
    #     pickle.dump(history.history, file)  # save history of training

    # # plots training history
    # plotter(history_file)  # plot accuracy and loss graphs

    # writes output to video
    video_write(model1,1)  # write model 1 predictions to video
    # video_write(model2,2)  # write model 2 predictions to video

    # # prepares data for svm training
    # svm_train_data1 = np.array([i[0] for i in data1])  # images from data1
    # svm_train_labels1 = np.array([i[1] for i in data1])  # labels from data1
    # svm_train_data2 = np.array([i[0] for i in data2])  # images from data2
    # svm_train_labels2 = np.array([i[1] for i in data2])  # labels from data2

    # # trains svm models
    # svm_model1 = svm_train(svm_train_data1, svm_train_labels1, model_name="svm_model1.pkl")
    # svm_model2 = svm_train(svm_train_data2, svm_train_labels2, model_name="svm_model2.pkl")

    # # loads svm models and make predictions
    # loaded_svm_model1 = load_svm_model(model_name="svm_model1.pkl")
    # svm_predictions_test1 = svm_predict(loaded_svm_model1, test_images1)
    # print("SVM 1 Predictions on test data1:", svm_predictions_test1)  # print svm model 1 predictions

    # loaded_svm_model2 = load_svm_model(model_name="svm_model2.pkl")
    # svm_predictions_test2 = svm_predict(loaded_svm_model2, test_images2)
    # print("SVM 2 Predictions on test data2:", svm_predictions_test2)  # print svm model 2 predictions

    # # writes classification answer
    # video_write(svm_model1,3)  # write model 1 predictions to video
    # video_write(svm_model2,4)  # write model 2 predictions to video
