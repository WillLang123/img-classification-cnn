import numpy as np
import constants as CONST
import pickle
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def SVMTrain(data, name):
    images = np.array([i[0] for i in data])  # get images
    labels = np.array([i[1] for i in data])  # get labels
    labels = np.argmax(labels, axis=1)  # Convert encoded labels to class labels (0 or 1)

    # Flatten images for SVM
    flattenedImages = images.reshape(-1, CONST.IMAGESIZE * CONST.IMAGESIZE * 3)

    # Split data into train and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(flattenedImages, labels, test_size=0.2, random_state=42)

    # Train SVM model
    print("making svm")
    svm = LinearSVC()  # linear SVM
    print("training svm")
    svm.fit(xTrain, yTrain)

    # Make predictions on test set
    print("fixing SVM")
    yPred = svm.predict(xTest)
    print("SVM Model Performance on Test Data:")
    print("Accuracy: ", accuracy_score(yTest, yPred))  # Print accuracy
    print("Classification Report:")
    print(classification_report(yTest, yPred))  # Print classification report

    # Save the trained SVM model
    pickle.dump(svm, open(name, 'wb'))
    print(f"SVM Model saved to {name}")  # Notify that model is saved


# loads the saved SVM model
def loadSVMModel(name):
    try:
        svm = pickle.load(open(name, 'rb'))
        print(f"SVM Model loaded successfully from {name}")
        return svm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# makes predictions with SVM model
def SVMPredict(model, images):
    flattenedImages = images.reshape(-1, CONST.IMAGESIZE * CONST.IMAGESIZE * 3)  # flatten images
    if(model is not None):
        return model.predict(flattenedImages)  # return predictions
    else:
        print('No model')
