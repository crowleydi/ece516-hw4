import numpy as np
import time

import keras
from keras.models import Model
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

def nested_cv(X, y,
              inner_cv, outer_cv, 
              DNN, param_grid,
              tr_batch_size=1, tr_epochs=10):
    """
    This function performs nested cross validation and returns
    and saves best model.
    Args:
        X (np array)    : Training numpy array.
                           X[0] gives first sample
                           X[r] gives range of values in r.
        y (np array)    : A numpy vector string the labels corresponding to X.
        inner_cv        : inner loop cross validation split
        outer_cv        : outer loop cross validation split
        DNN             : Keras model to train
        param_grid      : Hyper parameters values to optimize
        tr_batch_size   : Number of X-Samples per iteration
        tr_epochs       : Number of times to train using the same sample
                            size as a training data
    """
    train_history = []
    all_best_params = []
    all_test_loss = []
    all_test_metric = [] 
    # for each split of the data in the outer cross-validation
    # (split method returns indices of training and test parts)
    #
    out_split_idx = 1
    num_out_splits = outer_cv.get_n_splits(X, y)
    for training_samples, test_samples in outer_cv.split(X, y):
        print("Outer split: " + str(out_split_idx) + "/" + str(num_out_splits))
        out_split_idx+=1
        # find best parameter using inner cross-validation
        best_params = {}
        best_metric = -np.inf
        # iterate over parameters
        
        for par_idx, parameters in enumerate(param_grid):
            print("\tParameter : " + str(par_idx+1) + "/" + str(len(param_grid)))
            print("\t",parameters)
            # accumulate score over inner splits
            cv_loss = []
            cv_metric = []
            # iterate over inner cross-validation
            in_split_idx = 1
            num_in_splits = inner_cv.get_n_splits(X[training_samples], 
                                                  y[training_samples])
            in_start_time = time.time()
            for inner_train, inner_test in inner_cv.split(
                   X[training_samples], y[training_samples]):
                # build classifier given parameters and training data
                keras.backend.clear_session()
                dnn_inst = DNN(**parameters)
                keras_model = dnn_inst.model_
                in_start_time = time.time()
                keras_model.fit(X[inner_train], y[inner_train],
                                epochs=tr_epochs,
                                validation_split=0.2,
                                batch_size=tr_batch_size,
                                verbose=0)

                in_end_time = time.time()
                in_time_taken = in_end_time - in_start_time
                print("\t\tInner split: " + str(in_split_idx) + 
                      "/" + str(num_in_splits) + ", " + 
                      str(round(in_end_time - in_start_time)) + " sec")
                in_split_idx += 1

                # evaluate on inner test set
                loss, metric = keras_model.evaluate(X[inner_test], y[inner_test],verbose=0)
                cv_loss.append(loss)
                cv_metric.append(metric)
                   
     
            # compute mean score over inner folds
            # for a single combination of parameters.
            mean_loss = np.mean(cv_loss)
            mean_metric = np.mean(cv_metric)
            print("\t\tMean loss :" + str(mean_loss))
            print("\t\tMean Metric:" + str(mean_metric)) 
            if mean_metric > best_metric:
                # if better than so far, remember parameters
                best_metric  = mean_metric
                best_loss    = mean_loss
                best_params  = parameters
                

            # Save training results
            param_history = {
                "cv_loss"               : cv_loss,
                "cv_metric"             : cv_metric,
                "cv_params"             : parameters,
                "mean_loss"             : mean_loss,
                "mean_metric"           : mean_metric,
                "train_best_mean_loss"  : best_loss,
                "train_best_mean_metric": best_metric,
            }
            #print(param_history)
            train_history.append(param_history)


        # Build classifier on best parameters using outer training set
        # This is done over all parameters evaluated through a single
        # outer fold and all inner folds.
        # Fitting with best parameters and testing on
        # outer fold
        print("\tTraining on best prameters")
        best_dnn = DNN(**best_params)
        keras_model = best_dnn.model_
        keras_model.fit(X[training_samples], y[training_samples],epochs=tr_epochs,batch_size=tr_batch_size,verbose=0)
        test_loss, test_metric = keras_model.evaluate(X[test_samples], y[test_samples], verbose=0)
        print("\tLoss:" + str(test_loss))
        print("\tMetric:" + str(test_metric))
        # Outer loop test results:
        all_best_params.append(best_params)
        all_test_loss.append(test_loss)
        all_test_metric.append(test_metric) 

    train_results = {
        "train_history": train_history,
        "test_metric"  : all_test_metric,
        "best_params"  : all_best_params,
        "test_loss"    : all_test_loss
    }
    return train_results
    


