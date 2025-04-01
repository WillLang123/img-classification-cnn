import numpy as np
import pickle
import os
import cv2
import copy
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.svm import LinearSVC
import constants as CONST
from dataprep import prepData
from cnn import getCNNModel
from svm import SVMPredict, loadSVMModel, SVMTrain

def processImage(directory, imagePath):
    path = os.path.join(directory, imagePath)
    image = cv2.imread(path)  # read image
    imageCopy = copy.deepcopy(image)  # copy image

    image = cv2.resize(image, (CONST.IMG_SIZE, CONST.IMG_SIZE))  # resize image
    normImage = image.astype('float') / 255.0  # normalize image
    return imageCopy, normImage  # return processed image

# writes predictions to video
def videoWrite(model, i):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # codec
    filename = "./prediction" + str(i) + ".mp4"
    out = cv2.VideoWriter(filename, fourcc, 1.0, (400, 400))  # output video

    prediction = {1: 'Dog', 0: 'Cat'}  # mapping for predictions

    font = cv2.FONT_HERSHEY_DUPLEX
    location = (20, 30)
    fontScale = 0.5
    fontColor = (255, 255, 255)
    lineType = 2

    DIR = CONST.TEST_DIR
    MAX = CONST.OUTPUT_SIZE
    imagePaths = os.listdir(DIR)
    imagePaths = imagePaths[:MAX]  # limit to 100 images
    count = 0

    for imagePath in imagePaths:
        image, normImage = processImage(DIR, imagePath)  # process image
        
        if isinstance(model, LinearSVC):  # If it's an SVM model
            # Flatten the image for SVM (SVM expects 2D input with shape (n_samples, n_features))
            image_flattened = normImage.reshape(-1, CONST.IMG_SIZE * CONST.IMG_SIZE * 3)
            pred = SVMPredict(model, image_flattened)  # Use SVM model prediction

            # For SVM: pred is a single class label (either 0 or 1)
            # SVM is a hard classifier, so we assume 100% confidence
            arg_max = pred  # Directly use the label as arg_max

            # For SVM, just show the label and assume 100% confidence
            s = prediction[int(arg_max[0])] + ' - ' + '100%'  # Always 100% for SVM
            
        elif isinstance(model, tf.keras.Model):  # If it's a CNN model (TensorFlow Keras Model)
            normImage = normImage.reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # reshape image to fit CNN
            pred = model.predict(normImage)  # Keras CNN prediction

            # For CNN: pred will be a probability vector, so we extract the class with the highest probability
            arg_max = np.argmax(pred, axis=1)  # Get the class with highest probability
            max_val = np.max(pred, axis=1)  # Get the highest probability

            # For CNN, show the label and the percentage from the max value
            s = prediction[arg_max[0]] + ' - ' + str(max_val[0] * 100) + '%'

        else:
            raise TypeError("Unknown model type: " + str(type(model)))
        
        # Add the prediction to the image
        cv2.putText(image, s, location, font, fontScale, fontColor, lineType)  # add text to image

        image = cv2.resize(image, (400, 400))  # resize image
        out.write(image)  # write to video
        
        count += 1
        print(count)  # print progress
    out.release()  # release video


# loads and preps images for models
if __name__ == "__main__":

    # loads and preps images for models
    data1 = prepData(CONST.TRAIN_DIR_1)  # loading images from train1
    data2 = prepData(CONST.TRAIN_DIR_2)  # loading images from train2

    # figures out data size and what goes where
    trainingSize = int(CONST.DATA_SIZE * CONST.SPLIT_RATIO)
    print('data1', len(data1), trainingSize)  # size of data1
    print('data2', len(data2), trainingSize)  # size of data2

    # sets up tensorboard callback to use for CNN model
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

    # splits the data into training and testing sets for both datasets
    trainData1 = data1[:trainingSize]  # training data from dataset 1
    trainImages1 = np.array([i[0] for i in trainData1]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize images
    trainLabels1 = np.array([i[1] for i in trainData1])  # labels for dataset 1

    trainData2 = data2[:trainingSize]  # training data from dataset 2
    trainImages2 = np.array([i[0] for i in trainData2]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize images
    trainLabels2 = np.array([i[1] for i in trainData2])  # labels for dataset 2

    # splits the data into test sets for both datasets
    testData1 = data1[trainingSize:]  # test data from dataset 1
    testImages1 = np.array([i[0] for i in testData1]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize test images
    testLabels1 = np.array([i[1] for i in testData1])  # test labels for dataset 1

    testData2 = data2[trainingSize:]  # test data from dataset 2
    testImages2 = np.array([i[0] for i in testData2]).reshape(-1, CONST.IMG_SIZE, CONST.IMG_SIZE, 3)  # resize test images
    testLabels2 = np.array([i[1] for i in testData2])  # test labels for dataset 2

    # gets model to use
    model1 = getCNNModel()  # get CNN model for dataset 1
    print('dataset 1 training started...')
    history = model1.fit(
        trainImages1, trainLabels1, batch_size=50, epochs=15, verbose=1,
        validation_data=(testImages1, testLabels1), callbacks=[tensorboard_callback]
    )
    print('dataset 1 training done...')

    # save history of training
    history_file = '1000_history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)


    # trains model 2
    model2 = getCNNModel()  # get CNN model for dataset 2
    print('dataset 2 training started...')
    history = model2.fit(
        trainImages2, trainLabels2, batch_size=50, epochs=15, verbose=1,
        validation_data=(testImages2, testLabels2), callbacks=[tensorboard_callback]
    )
    print('dataset 2 training done...')

    # save history of training
    history_file = '1000_history.pickle'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    # writes output to video
    videoWrite(model1,1)  # write model 1 predictions to video
    videoWrite(model2,2)  # write model 2 predictions to video

    # trains svm models
    SVMModel1 = SVMTrain(data1, model_name="svm_model1.pkl")
    SVMModel2 = SVMTrain(data2, model_name="svm_model2.pkl")

    # loads svm models and make predictions
    loadedSVMModel1 = loadSVMModel(model_name="svm_model1.pkl")
    SVMPredictions1 = SVMPredict(loadedSVMModel1, testImages1)
    print("SVM 1 Predictions on test data1:", SVMPredictions1)  # print svm model 1 predictions

    loadedSVMModel2 = loadSVMModel(model_name="svm_model2.pkl")
    SVMPredictions2 = SVMPredict(loadedSVMModel2, testImages2)
    print("SVM 2 Predictions on test data2:", SVMPredictions2)  # print svm model 2 predictions

    # writes classification answer
    videoWrite(loadedSVMModel1,3)  # write model 1 predictions to video
    videoWrite(loadedSVMModel2,4)  # write model 2 predictions to video
