import numpy as np
import scipy.io
import pandas as pd
import os
import time

import keras
from keras.models import Model
from keras.layers import Dense, Conv1D, MaxPool1D, Input, Flatten, MaxPooling1D, Dropout,Conv2D, MaxPooling2D
from keras.layers import Conv1D
from metrics import *

#######################################################################
# Dense network with one hidden layer: 737 ->128->1
def dense_1():
  input_layer = Input(shape = (737,), name = 'input_layer')
  # Embedding
  x = Dense(128, activation = 'relu')(input_layer)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

#######################################################################


#######################################################################
# Dense network with two hidden layers: 737 ->128->128->1
def dense_2():
  #Input layer

  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(128, activation = 'relu')(input_layer)
  x = Dense(128, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model
#######################################################################

#######################################################################
# Dense network with two hidden layers: 737 ->256->128->1
def dense_3():
  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(256, activation = 'relu')(input_layer)
  x = Dense(128, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

#######################################################################
# Dense network with two hidden layers: 737 ->128->64->1
def dense_4():
  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(128, activation = 'relu')(input_layer)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

# Dense network with two hidden layers: 737 ->64->32->1
def dense_5():
  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(64, activation = 'relu')(input_layer)
  x = Dense(32, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

#Same as 4 with dropout
def dense_6():
  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(128, activation = 'relu')(input_layer)
  x = Dropout(0.2)(x)
  x = Dense(64, activation = 'relu')(x)
  x = Dropout(0.2)(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def dense_7():
  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(256, activation = 'relu')(input_layer)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def dense_8():
  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(256, activation = 'relu')(input_layer)
  x = Dense(32, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def dense_9():
  input_layer = Input(shape = (737,), name = 'input_layer')
# Embedding
  x = Dense(256, activation = 'relu')(input_layer)
  x = Dense(64, activation = 'relu')(x)
  x = Dense(32, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model


#######################################################################

###################     Convolutional Networks    #####################

#######################################################################

def conv_dense_k3_f8(desc = 'conv_dense_'):
  input_layer = Input(shape = (736,1), name = 'input_layer')
  x = Conv1D(filters = 8, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  auxiliary_input = Input(shape=(1,), name='aux_input')
  x = keras.layers.concatenate([x, auxiliary_input])
  x = Dense(256, activation = 'relu')(x)
  x = Dense(128, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer, auxiliary_input], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model



def conv_dense_k3_f4(desc = 'conv_dense_'):
  input_layer = Input(shape = (736,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  auxiliary_input = Input(shape=(1,), name='aux_input')
  x = keras.layers.concatenate([x, auxiliary_input])
  x = Dense(256, activation = 'relu')(x)
  x = Dense(128, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer, auxiliary_input], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def conv_dense_k3_f4_f2(desc = 'conv_dense_'):
  input_layer = Input(shape = (736,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Conv1D(filters = 2, kernel_size = 3, activation='relu', padding = 'same')(x)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  auxiliary_input = Input(shape=(1,), name='aux_input')
  x = keras.layers.concatenate([x, auxiliary_input])
  x = Dense(256, activation = 'relu')(x)
  x = Dense(128, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer, auxiliary_input], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def conv_dense_k3_f4_dense_4(desc = 'conv_dense_'):
  input_layer = Input(shape = (737,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  x = Dense(128, activation = 'relu')(x)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def conv_dense_k3_f4_v2(desc = 'conv_dense_'):
  input_layer = Input(shape = (737,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  x = Dense(256, activation = 'relu')(x)
  x = Dense(128, activation = 'relu')(x)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def conv_dense_k3_f4_v3(desc = 'conv_dense_'):
  input_layer = Input(shape = (737,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  x = Dense(256, activation = 'relu')(x)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def conv_dense_k3_f4_v4(desc = 'conv_dense_'):
  input_layer = Input(shape = (737,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  x = Dense(512, activation = 'relu')(x)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def conv_dense_k3_f4_v5(desc = 'conv_dense_'):
  input_layer = Input(shape = (737,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 3, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  x = Dense(256, activation = 'relu')(x)
  x = Dense(32, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model

def conv_dense_k3_f4_v6(desc = 'conv_dense_'):
  input_layer = Input(shape = (737,1), name = 'input_layer')
  x = Conv1D(filters = 4, kernel_size = 5, activation='relu', padding = 'same')(input_layer)
  x = MaxPooling1D(pool_size=2, padding='valid', strides = 2)(x)
  x = Flatten()(x)
  x = Dense(256, activation = 'relu')(x)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',loss='mse')
  return model
#######################################################################
#######################################################################
#######################################################################
#######################################################################



#######################################################################

###################         2D models            #####################

#######################################################################
def map_4conv_8421():
  input_layer = Input(shape = (222, 772,2), name = 'input_layer')
  x = Conv2D(8, kernel_size=(3,3), 
          activation='relu', padding= 'same')(input_layer)
  x = MaxPooling2D(pool_size=(4, 4), strides=2, padding='valid')(x)
  x = Conv2D(4, kernel_size=(3,3), 
          activation='relu', padding= 'same')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
  x = Conv2D(2, kernel_size=(3,3), 
          activation='relu', padding= 'same')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
  x = Conv2D(1, kernel_size=(3,3), 
          activation='relu', padding= 'same')(x)
  x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
  x = Flatten()(x)
  x = Dense(32, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',
                loss='mse')
  return model


def map_3conv_4_8_16():
  input_layer = Input(shape = (222, 772,2), name = 'input_layer')
  x = Conv2D(4, kernel_size=(3,3), 
        activation='relu', padding= 'same')(input_layer)
  x = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(x)
  x = Conv2D(8, kernel_size=(3,3), 
        activation='relu', padding= 'same')(x)
  x = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(x)
  x = Conv2D(16, kernel_size=(3,3), 
        activation='relu', padding= 'same')(x)
  x = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(x)
  x = Flatten()(x)
  x = Dense(256, activation = 'relu')(x)
  x = Dense(64, activation = 'relu')(x)
  out = Dense(1)(x)
  model = Model(inputs = [input_layer], outputs  = [out])
  model.compile(optimizer='Adam',
              loss='mse')
  return model

