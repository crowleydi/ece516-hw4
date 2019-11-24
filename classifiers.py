import numpy as np

import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import binary_crossentropy

class BaseDNN:
    """
    SampleDNN simply returns a keras model. It does not
    replace the functional api.
    """

    def __init__(self, 
                 input_size, 
                 num_first_filters,
                 num_conv_nets,
                 kernel_size = (2,2,2), 
                 pool_size   =(2,2,2)):
        """
        Class having methods and techniques to explore DNN
        classifiers using keras library. 
        
        The model is saved in <object name>.model_

        Args:
            input_size (tuple)  : (num_frames, num_rows, num_cols, num_channels)
                1. writing no writing uv np arrays = (89,  100, 100, 2)
                2. talking no talking uv np arrays = (299, 100, 100, 2)
                3. writing no writing gray np arrays = (90,  100, 100, 1)
                4. talking no talking gray np arrays = (300,  100, 100, 1)

            num_first_filters   : Number of first layer filters

            num_conv_nets (int) : Number of ConvNets

            kernel_size (tuple) : (num_frames, num_rows, num_cols)
                                  Default = (3,3,3)
        """
        self.input_size_        = input_size
        self.num_first_filters_ = num_first_filters 
        self.num_conv_nets_     = num_conv_nets
        self.kernel_size_       = kernel_size
        self.pool_size_         = pool_size
        self.model_             = self.build()

    def __repr__(self):
        """
        Provides internal information for developers
        """
        info = """
            The SampleDNN class provides examples for you to
            build your own class.
        """
        

    def __str__(self):
        """
        Prints model summary after the build method.
        
        Example:
            print(name_of_the_object)
        """
        info  = "Architecture:\n"+\
                "    Input size = " + str(self.input_size_)+\
                "    Number of first filters = " + str(self.num_first_filters_)+\
                "    Number of ConvNets = " + str(self.num_conv_nets_)
        return info


class SampleDNN(BaseDNN):
    def build(self):
        """
        Builds keras model using parameters provided.
        """
        input_layer = Input(self.input_size_)
        pool_layer = input_layer
  
        for i in range(self.num_conv_nets_):
            conv_layer = Conv3D(filters     = self.num_first_filters_, 
                                kernel_size= self.kernel_size_, 
                                activation='relu',
                                data_format = "channels_last")(pool_layer)
            pool_layer = MaxPool3D(pool_size= self.pool_size_,
                                   data_format="channels_last")(conv_layer)

            pool_layer = BatchNormalization()(pool_layer)

        flatten_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        #dense_layer2 = Dense(units=32, activation='relu')(dense_layer1)
        #dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=1, activation='sigmoid')(dense_layer1)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss=binary_crossentropy,
                optimizer='sgd', metrics=['acc'])
        
        return self.model

class PyramidDNN(BaseDNN):
    def build(self):
        """
        Builds keras model using parameters provided.
        """
        input_layer = Input(self.input_size_)
        pool_layer = input_layer
  
        N = self.num_first_filters_
        for i in range(self.num_conv_nets_):
            conv_layer = Conv3D(filters = N, 
                                kernel_size= self.kernel_size_, 
                                activation='relu',
                                data_format = "channels_last")(pool_layer)
            pool_layer = MaxPool3D(pool_size= self.pool_size_,
                                   data_format="channels_last")(conv_layer)

            pool_layer = BatchNormalization()(pool_layer)
            N = N // 2

        flatten_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        #dense_layer2 = Dense(units=32, activation='relu')(dense_layer1)
        #dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=1, activation='sigmoid')(dense_layer1)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss=binary_crossentropy,
                optimizer='sgd', metrics=['acc'])
        
        return self.model

class InvPyramidDNN(BaseDNN):
    def build(self):
        """
        Builds keras model using parameters provided.
        """
        input_layer = Input(self.input_size_)
        pool_layer = input_layer
  
        N = self.num_first_filters_
        for i in range(self.num_conv_nets_):
            conv_layer = Conv3D(filters = N, 
                                kernel_size= self.kernel_size_, 
                                activation='relu',
                                data_format = "channels_last")(pool_layer)
            pool_layer = MaxPool3D(pool_size= self.pool_size_,
                                   data_format="channels_last")(conv_layer)

            pool_layer = BatchNormalization()(pool_layer)
            N = N * 2

        flatten_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        #dense_layer2 = Dense(units=32, activation='relu')(dense_layer1)
        #dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=1, activation='sigmoid')(dense_layer1)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss=binary_crossentropy,
                optimizer='sgd', metrics=['acc'])
        
        return self.model

class SingleDNN(BaseDNN):
    def build(self):
        """
        Builds keras model using parameters provided.
        """
        input_layer = Input(self.input_size_)
        pool_layer = input_layer
  
        N = self.num_first_filters_
        conv_layer = Conv3D(filters = N, 
                                kernel_size= self.kernel_size_, 
                                activation='relu',
                                data_format = "channels_last")(pool_layer)
        pool_layer = MaxPool3D(pool_size= self.pool_size_,
                                   data_format="channels_last")(conv_layer)

        pool_layer = BatchNormalization()(pool_layer)

        flatten_layer = Flatten()(pool_layer)

        dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        #dense_layer2 = Dense(units=32, activation='relu')(dense_layer1)
        #dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=1, activation='sigmoid')(dense_layer1)
        
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss=binary_crossentropy,
                optimizer='sgd', metrics=['acc'])
        
        return self.model

