import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from data_prep import prep_and_load_data
import constants as CONST
import pickle
import os

def svm_train(data, model_name="svm_model.pkl"):
    images = np.array([i[0] for i in data])  # get images
    labels = np.array([i[1] for i in data])  # get labels

    # Convert from one-hot encoding to scalar labels
    labels = np.argmax(labels, axis=1)  # Convert one-hot encoded labels to class labels (0 or 1)

    # Flatten images for SVM
    images_flattened = images.reshape(-1, CONST.IMG_SIZE * CONST.IMG_SIZE * 3)  # flatten image for SVM input

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(images_flattened, labels, test_size=0.2, random_state=42)

    # Train SVM model
    print("making svm")
    svm_classifier = SVC(kernel='linear', probability=True)  # linear SVM
    print("training svm")
    svm_classifier.fit(X_train, y_train)

    # Make predictions on test set
    print("fixing SVM")
    y_pred = svm_classifier.predict(X_test)
    print("SVM Model Performance on Test Data:")
    print("Accuracy: ", accuracy_score(y_test, y_pred))  # Print accuracy
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Print classification report

    # Save the trained SVM model
    with open(model_name, 'wb') as model_file:
        pickle.dump(svm_classifier, model_file)
    print(f"SVM Model saved to {model_name}")  # Notify that model is saved


# loads the saved SVM model
def load_svm_model(model_name="svm_model.pkl"):
    try:
        with open(model_name, 'rb') as model_file:
            svm_classifier = pickle.load(model_file)
        print(f"SVM Model loaded successfully from {model_name}")
        return svm_classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# makes predictions with SVM model
def svm_predict(model, images):
    images_flattened = images.reshape(-1, CONST.IMG_SIZE * CONST.IMG_SIZE * 3)  # flatten images
    if(model is not None):
        return model.predict(images_flattened)  # return predictions
    else:
        print('No model')
