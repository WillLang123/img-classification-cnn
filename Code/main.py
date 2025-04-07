import numpy as np
import pickle
import os
import cv2
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from sklearn.svm import LinearSVC
import constants as CONST
from dataprep import prepData, processImage
from cnn import getCNNModel
from svm import SVMPredict, loadSVMModel, SVMTrain

# writes predictions to video
def videoWrite(model, i):
    out = cv2.VideoWriter(("./prediction" + str(i) + ".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (400, 400))  # output video
    prediction = {1: 'dog', 0: 'cat'}
    imagePaths = os.listdir(CONST.TESTING)
    imagePaths = imagePaths[:CONST.OUTPUTSIZE]  # limit to 100 images
    count = 0

    for imagePath in imagePaths:
        image, normImage = processImage(CONST.TESTING, imagePath)  # process image
        
        if isinstance(model, LinearSVC):  # If it's an SVM model
            # Flatten the image for SVM (SVM expects 2D input with shape (n_samples, n_features))
            flattenedImages = normImage.reshape(-1, CONST.IMAGESIZE * CONST.IMAGESIZE * 3)
            pred = SVMPredict(model, flattenedImages)  # Use SVM model prediction

            # For SVM: pred is a single class label (either 0 or 1)
            # SVM is a hard classifier, so we assume 100% confidence
            arg_max = pred  # Directly use the label as arg_max

            # For SVM, just show the label and assume 100% confidence
            s = prediction[int(arg_max[0])] + ' - ' + '100%'  # Always 100% for SVM
            
        elif isinstance(model, tf.keras.Model):  # If it's a CNN model (TensorFlow Keras Model)
            normImage = normImage.reshape(-1, CONST.IMAGESIZE, CONST.IMAGESIZE, 3)  # reshape image to fit CNN
            pred = model.predict(normImage)  # Keras CNN prediction

            # For CNN: pred will be a probability vector, so we extract the class with the highest probability
            arg_max = np.argmax(pred, axis=1)  # Get the class with highest probability
            max_val = np.max(pred, axis=1)  # Get the highest probability

            # For CNN, show the label and the percentage from the max value
            s = prediction[arg_max[0]] + ' - ' + str(max_val[0] * 100) + '%'

        else:
            raise TypeError("Unknown model type: " + str(type(model)))
        
        # Add the prediction to the image
        white = [255,255,255]
        textPlace = [20,20]
        fontSize = .5
        outputDim = [400,400]
        cv2.putText(image, s, textPlace, cv2.FONT_HERSHEY_SIMPLEX, fontSize, white)
        image = cv2.resize(image, outputDim)  # resize image
        out.write(image)  # write to video
        count += 1
        print(count)  # print progress
    out.release()  # release video


# loads and preps images for models
if __name__ == "__main__":

    # loads and preps images for models
    data1 = prepData(CONST.TRAINING1)  # loading images from train1
    data2 = prepData(CONST.TRAINING2)  # loading images from train2

    # figures out data size and what goes where
    trainingSize = int(CONST.DATASIZE * CONST.SPLITRATIO)
    # print('data1', len(data1), trainingSize)  # size of data1
    # print('data2', len(data2), trainingSize)  # size of data2

    # sets up tensorboard callback to use for CNN model
    tfCallback = TensorBoard(log_dir='./logs', histogram_freq=1)

    # splits the data into training and testing sets for both datasets
    trainData1 = data1[:trainingSize]  # training data from dataset 1
    trainImages1, trainLabels1 = [], []
    for item in trainData1:
        trainImages1.append(item[0])
        trainLabels1.append(item[1])
    trainImages1 = np.array(trainImages1).reshape(-1, CONST.IMAGESIZE, CONST.IMAGESIZE, 3)  # resize images
    trainLabels1 = np.array(trainLabels1) #need to make numpy to fit input dimensions of x

    trainData2 = data2[:trainingSize]  # training data from dataset 2
    trainImages2, trainLabels2 = [], []
    for item in trainData2:
        trainImages2.append(item[0])
        trainLabels2.append(item[1])
    trainImages2 = np.array(trainImages2).reshape(-1, CONST.IMAGESIZE, CONST.IMAGESIZE, 3)  # resize images
    trainLabels2 = np.array(trainLabels2) #need to make numpy to fit input dimensions of x

    # splits the data into test sets for both datasets
    testData1 = data1[trainingSize:]  # test data from dataset 1
    testImages1, testLabels1 = [], []
    for item in testData1:
        testImages1.append(item[0])
        testLabels1.append(item[1])
    testImages1 = np.array(testImages1).reshape(-1, CONST.IMAGESIZE, CONST.IMAGESIZE, 3)  # resize test images
    testLabels1 = np.array(testLabels1) #need to make numpy to fit input dimensions of x

    testData2 = data2[trainingSize:]  # test data from dataset 2
    testImages2, testLabels2 = [], []
    for item in testData2:
        testImages2.append(item[0])
        testLabels2.append(item[1])
    testImages2 = np.array(testImages2).reshape(-1, CONST.IMAGESIZE, CONST.IMAGESIZE, 3)  # resize test images
    testLabels2 = np.array(testLabels2) #need to make numpy to fit input dimensions of x

    # gets model to use
    model1 = getCNNModel()  # get CNN model for dataset 1
    print('dataset 1 training started...')
    history = model1.fit(
        trainImages1, trainLabels1, batch_size=50, epochs=15, verbose=1,
        validation_data=(testImages1, testLabels1), callbacks=[tfCallback]
    )
    print('dataset 1 training done...')

    # save history of training
    historyFile1 = '6000History1.pickle'
    pickle.dump(history.history, open(historyFile1, 'wb'))


    # trains model 2
    model2 = getCNNModel()  # get CNN model for dataset 2
    print('dataset 2 training started...')
    history = model2.fit(
        trainImages2, trainLabels2, batch_size=50, epochs=15, verbose=1,
        validation_data=(testImages2, testLabels2), callbacks=[tfCallback]
    )
    print('dataset 2 training done...')

    # save history of training
    historyFile2 = '6000History2.pickle'
    pickle.dump(history.history, open(historyFile2, 'wb'))

    # writes output to video
    videoWrite(model1,1)  # write model 1 predictions to video
    videoWrite(model2,2)  # write model 2 predictions to video

    # trains svm models
    SVMModel1 = SVMTrain(data1, "svm1.pkl")
    SVMModel2 = SVMTrain(data2, "svm2.pkl")

    # loads svm models and make predictions
    loadedSVMModel1 = loadSVMModel(name="svm1.pkl")
    SVMPredictions1 = SVMPredict(loadedSVMModel1, testImages1)
    # print("SVM 1 Predictions on test data1:", SVMPredictions1)  # print svm model 1 predictions

    loadedSVMModel2 = loadSVMModel(name="svm2.pkl")
    SVMPredictions2 = SVMPredict(loadedSVMModel2, testImages2)
    # print("SVM 2 Predictions on test data2:", SVMPredictions2)  # print svm model 2 predictions

    # writes classification answer
    videoWrite(loadedSVMModel1,3)  # write model 1 predictions to video
    videoWrite(loadedSVMModel2,4)  # write model 2 predictions to video
