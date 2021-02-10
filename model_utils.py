from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import model_from_json
import os.path 

# define model
def define_model():
    # model = Sequential()

    # # 1st stage
    # model.add(Conv2D(32, 3, input_shape=(48, 48, 1), padding='same',
    #                 activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(32, 3, padding='same',
    #                 activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2, 2), strides=2))

    # # 2nd stage
    # model.add(Conv2D(64, 3, padding='same',
    #                 activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, 3, padding='same',
    #                 activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    # model.add(Dropout(0.25))

    # # 3rd stage
    # model.add(Conv2D(128, 3, padding='same',
    #                 activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(128, 3, padding='same',
    #                 activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    # model.add(Dropout(0.25))

    # # FC layers
    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # model.add(Dense(256))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # model.add(Dense(256))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    # model.add(Dense(7))
    # model.add(Activation('softmax'))


    # # compile the model
    # model.compile(loss='categorical_crossentropy',
    #           optimizer='adam', metrics=['accuracy'])
    
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    return model 

# load model weights
def model_weights(model):
    model.load_weights('fer.h5')
    return model

