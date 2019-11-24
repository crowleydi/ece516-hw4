import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import pdb
import cv2
import numpy as np
import pprint
import pickle

import keras

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

all_batches = 4
all_epochs  = 50

from classifiers import SampleDNN, PyramidDNN, InvPyramidDNN, SingleDNN

dnn = SampleDNN
X_train_ori_name = './nparrays/wnw/train_gray_X_100.npy'
y_train_ori_name = './nparrays/wnw/train_gray_y_100.npy'
X_test_name = './nparrays/wnw/test_gray_X_200.npy'
y_test_name = './nparrays/wnw/test_gray_y_200.npy'
model_name  = 'best_model_gray_100_bn.h5'

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
        keras_model.fit(X[training_samples], y[training_samples],epochs=all_epochs,batch_size=all_batches,verbose=0)
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
    


X_train_ori = np.load(X_train_ori_name)
y_train_ori = np.load(y_train_ori_name)

# `in_size` for 
#     1. writing no writing uv np arrays = (89,  100, 100, 2)
#     2. talking no talking uv np arrays = (299, 100, 100, 2)
#     3. writing no writing gray np arrays = (90,  100, 100, 1)
#     4. talking no talking gray np arrays = (300,  100, 100, 1)
in_size       = [X_train_ori.shape[1:]]
num_first_fil = [2,4]
num_convnets  = [2,3]
param_grid = {"input_size": in_size,
              "num_first_filters": num_first_fil,
              "num_conv_nets":num_convnets}

train_results = nested_cv(X_train_ori, y_train_ori,   StratifiedKFold(3), 
                          StratifiedKFold(3), dnn, 
                          ParameterGrid(param_grid),
                          tr_batch_size=all_batches,
                          tr_epochs=all_epochs)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(train_results)
print("\n\n")
scores = np.array(train_results["test_metric"])
print("Cross-validation scores: {}".format(scores)) 
stats_list = (scores.min(), scores.max(), scores.mean())
print("Min value = {:0.2f}, Max value = {:0.2f}, Mean = {:0.2f}".format(*stats_list))



# Split 80% for training and 20% for testing
X_train, X_val, y_train, y_val = train_test_split(X_train_ori, y_train_ori, test_size=0.2)
best_of_all_metric = -np.inf 
tst = train_results["best_params"]
for i in range(len(tst)):

    # Train on 80%
    keras.backend.clear_session()
    best_dnn = dnn(**tst[i])
    keras_model = best_dnn.model_
    keras_model.fit(X_train, y_train, epochs=all_epochs, batch_size=all_batches,verbose=0)

    loss, metric = keras_model.evaluate(X_val, y_val)
    
    # Determine the best of the best
    if metric > best_of_all_metric:
        best_of_all_params = tst[i]
        best_of_all_loss   = loss
        best_of_all_metric = metric

print("\n============== BEST OF ALL =============")
print(best_of_all_params)
print("Best of all loss: "   + str(best_of_all_loss))
print("Best of all metric: " + str(best_of_all_metric))
print("========================================\n")


print("Training on entire training set with best of all parameters")
keras.backend.clear_session()
best_dnn = dnn(**best_of_all_params)
print(best_dnn)

keras_model = best_dnn.model_
history = keras_model.fit(X_train_ori, y_train_ori, epochs=all_epochs, batch_size=all_batches,verbose=0)
keras_model.save(model_name)
with open(model_name + ".pickle", 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
# Delete the existing model.
del keras_model



# TESTING PHASE
X_test = np.load(X_test_name)
y_test = np.load(y_test_name)
model  = load_model(model_name)
y_pred = 1*(model.predict(X_test) > 0.5)
y_pred = y_pred.flatten()
conf_matrix = confusion_matrix(y_test,y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\n Accuracy on test set: " + str(accuracy))
print(conf_matrix)

# Testing on training to check learning
y_pred      = 1*(model.predict(X_train) > 0.5)
y_pred      = y_pred.flatten()
conf_matrix = confusion_matrix(y_train,y_pred)
accuracy    = accuracy_score(y_train, y_pred)
print("\nAccuracy on train set: " + str(accuracy)) 
print(conf_matrix)
