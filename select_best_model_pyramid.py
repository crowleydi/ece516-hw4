import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import time
import pdb
import cv2
import numpy as np
import pprint
import pickle

import keras
from keras.models import load_model

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

all_batches = 4
all_epochs  = 50

from classifiers import PyramidDNN, InvPyramidDNN, SingleDNN, nested_cv
dnn = PyramidDNN

X_train_ori_name = './nparrays/wnw/train_gray_X_100.npy'
y_train_ori_name = './nparrays/wnw/train_gray_y_100.npy'
X_test_name = './nparrays/wnw/test_gray_X_200.npy'
y_test_name = './nparrays/wnw/test_gray_y_200.npy'
model_name  = 'best_model_wnw_pyramid.h5'

X_train_ori = np.load(X_train_ori_name)
y_train_ori = np.load(y_train_ori_name)

# `in_size` for 
#     1. writing no writing uv np arrays = (89,  100, 100, 2)
#     2. talking no talking uv np arrays = (299, 100, 100, 2)
#     3. writing no writing gray np arrays = (90,  100, 100, 1)
#     4. talking no talking gray np arrays = (300,  100, 100, 1)
in_size       = [X_train_ori.shape[1:]]
num_first_fil = [8,4]
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
