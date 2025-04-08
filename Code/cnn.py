import tensorflow
import constants as CONST
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Dropout
from keras import activations, optimizers, losses, metrics

# builds the CNN model
def getCNNModel():
    inputDim = (CONST.IMAGESIZE, CONST.IMAGESIZE, 3)
    model = tensorflow.keras.Sequential(
        [
            #makes model
            Input(shape=inputDim),
            Conv2D(32, (3, 3), activation=activations.relu), #adds convolution layer
            MaxPooling2D((2,2)), #maxpooling layer
            BatchNormalization(), #normalize
            
            Conv2D(64, (3,3), activation=activations.relu), #convolution layer
            MaxPooling2D((2,2)), #maxpooling layer
            BatchNormalization(), #normalize
            
            Conv2D(96, (3,3), activation=activations.relu), #convolution layer
            MaxPooling2D((2,2)), #maxpooling layer
            BatchNormalization(), #normalize
            
            Conv2D(96, (3,3), activation=activations.relu), #convolution layer
            MaxPooling2D((2,2)), #maxpooling layer
            BatchNormalization(), # normalize
            Dropout(0.2), # use dropout to try and fix overfitting
            
            Conv2D(64, (3,3), activation=activations.relu), #convolution layer
            MaxPooling2D((2,2)), #maxpooling layer
            BatchNormalization(), #  normalize
            Dropout(0.2), # dropout again to fight overfitting
            
            Flatten(), # flattern convoluted image to 2d
            Dense(256, activation=activations.relu), # classification using weights
            Dropout(0.2), # tries to fix overfitting
            Dense(128, activation=activations.relu), # classification using weights
            Dropout(0.3), # tries to fix overfiting
            Dense(2, activation = activations.sigmoid) # classification using weights
        ]
    ) 
    
    model.compile(
        loss=losses.BinaryCrossentropy(), 
        optimizer=optimizers.Adam(learning_rate=1e-3), 
        metrics = [metrics.BinaryAccuracy(), metrics.FalsePositives()])
    print('model prepared...')
    return model
    # How I figured out the model layers:
    # model.add(Input(shape=inputDim))
    # model.add(Conv2D(64, (3,3), activation=activations.relu))
    # model.summary() #debugging