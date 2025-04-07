import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Dropout
import constants as CONST

# builds the CNN model
def getCNNModel():
    model = tf.keras.Sequential(
        
    ) #makes model
    model.add(Input(shape=(CONST.IMAGESIZE, CONST.IMAGESIZE, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu')) #adds convolution layer
    model.add(MaxPooling2D((2,2))) #maxpooling layer
    model.add(BatchNormalization()) #normalize
    # model.summary() #debugging
    model.add(Conv2D(64, (3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D((2,2))) #maxpooling layer
    model.add(BatchNormalization()) #normalize
    
    model.add(Conv2D(96, (3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D((2,2))) #maxpooling layer
    model.add(BatchNormalization()) #normalize
    
    model.add(Conv2D(96, (3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D((2,2))) #maxpooling layer
    model.add(BatchNormalization()) # normalize
    model.add(Dropout(0.2)) # use dropout to try and fix overfitting
    
    model.add(Conv2D(64, (3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D((2,2))) #maxpooling layer
    model.add(BatchNormalization()) #  normalize
    model.add(Dropout(0.2)) # dropout again to fight overfitting
    
    model.add(Flatten()) # flattern convoluted image to 2d
    model.add(Dense(256, activation='relu')) # classification using weights
    model.add(Dropout(0.2)) # tries to fix overfitting
    model.add(Dense(128, activation='relu')) # classification using weights
    model.add(Dropout(0.3)) # tries to fix overfiting
    model.add(Dense(2, activation = 'sigmoid')) # classification using weights

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print('model prepared...')
    return model