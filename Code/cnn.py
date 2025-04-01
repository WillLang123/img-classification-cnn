import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import constants as CONST

# builds the CNN model
def getCNNModel():
    model = tf.keras.Sequential()#Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(CONST.IMG_SIZE, CONST.IMG_SIZE, 3))) #convolution layer
    model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling layer
    model.add(BatchNormalization()) #normalize
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling layer
    model.add(BatchNormalization()) #normalize
    
    
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling layer
    model.add(BatchNormalization()) #normalize
    
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling layer
    model.add(BatchNormalization()) # normalize
    model.add(Dropout(0.2)) # use dropout to try and fix overfitting
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) #convolution layer
    model.add(MaxPooling2D(pool_size=(2,2))) #maxpooling layer
    model.add(BatchNormalization()) #  normalize
    model.add(Dropout(0.2)) # dropout again to fight overfitting
    
    model.add(Flatten()) # flattern convoluted image to 2d
    model.add(Dense(256, activation='relu')) # classification using weights
    model.add(Dropout(0.2)) # tries to fix overfitting
    model.add(Dense(128, activation='relu')) # classification using weights
    model.add(Dropout(0.3)) # tries to fix overfiting
    model.add(Dense(2, activation = 'softmax')) # classification using weights (might make sigmoid for better classification)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    print('model prepared...')
    return model