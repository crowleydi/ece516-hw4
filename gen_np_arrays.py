# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:07:52 2019

@author: shi
"""

import numpy as np
import cv2
import os
import random
from random import sample 
import math  as math
import pandas as pd
import pdb
import sys
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


SAVE_TALKING_GRAY = False
SAVE_TALKING_FLOW = False

SAVE_WRITING_GRAY = False
SAVE_WRITING_FLOW = True


def videosdir_2_gray_np(fpath, y_label):
    """
    Converts list of videos to 4D numpy array. The numpy array
    has gray scale frames stored.
        (file_idx, frame_num, num_rows, num_cols, num_channels)
    
    Args:
        fpath (str)      : Directory of the videos
        y_label (int)     : Label corresponding to files files passed.
                             It can take the following values {0,1}
                             0 = No activity
                             1 = Activity
    """
    # Getting trimmed video properties
    file_names            = os.listdir(fpath)
    vid                   = cv2.VideoCapture(fpath+file_names[0])
    num_frms              = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frm              = vid.read()
    num_rows, num_cols, _ = frm.shape
    num_channels          = 1

    # Creating ndarray
    X = np.ndarray((len(file_names), 
                    num_frms,
                    num_rows,
                    num_cols,
                    num_channels),
                   dtype="uint8")
    y = np.array([y_label]*len(file_names))
    
    # file loop
    for fidx,fname in enumerate(file_names):

        vid      = cv2.VideoCapture(fpath+fname)
        frm_num  = 0
        ret, frm = vid.read()

        # Video loop
        while vid.isOpened() and ret:
            gray_frm            = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
            X[fidx,frm_num,:,:,0] = gray_frm # 0 channel
            frm_num += 1
            ret, frm            = vid.read()
    
    return X, y


def videosdir_2_flow_np(fpath, y_label, num_samples):
    """
    Converts list of videos to 4D numpy array. The numpy array
    has gray scale frames stored.
        (file_idx, frame_num, num_rows, num_cols, num_channels)
    
    Args:
        fpath (str)      : Directory of the videos
        y_label (int)    : Label corresponding to files files passed.
                             It can take the following values {0,1}
                             0 = No activity
                             1 = Activity
        num_samples      : Number of samples to consider
    """
    # Getting trimmed video properties
    file_names            = os.listdir(fpath)
    file_names            = sample(file_names,num_samples)
    vid                   = cv2.VideoCapture(fpath+file_names[0])
    num_frms              = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frm              = vid.read()
    num_rows, num_cols, _ = frm.shape
    num_channels          = 2

    # Creating ndarray
    X = np.ndarray((len(file_names), 
                    num_frms-1,
                    num_rows,
                    num_cols,
                    num_channels),
                   dtype="float")
    y = np.array([y_label]*len(file_names))
    
    for fidx,fname in enumerate(file_names):
        print(fname)
        vid       = cv2.VideoCapture(fpath+fname)
        ret, frm0 = vid.read()
        gray_frm0 = cv2.cvtColor(frm0,cv2.COLOR_BGR2GRAY)
        ret, frm1             = vid.read()

        # Video loop
        flow_num  = 0
        while vid.isOpened() and ret:
            
            # Calculating flow
            gray_frm1  = cv2.cvtColor(frm1,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray_frm0, gray_frm1,None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            # Storing in X
            X[fidx,flow_num,:,:,:] = flow
            
            # New frame
            flow_num  += 1
            gray_frm0  = gray_frm1.copy()
            ret, frm1  = vid.read()
    
    return X, y


            
        

# Creating Talking and no talking Gray numpy arrays
if SAVE_TALKING_GRAY:
    t_path  = './videos/talking/'
    nt_path = './videos/no_talking/'

    X_talk, y_talk     = videosdir_2_gray_np(t_path,1)
    X_notalk, y_notalk = videosdir_2_gray_np(nt_path,0)
    X                  = np.concatenate((X_talk, X_notalk), axis=0)
    y                  = np.concatenate((y_talk, y_notalk), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=19)
    np.save('./nparrays/tnt/train_gray_X', X_train)
    np.save('./nparrays/tnt/train_gray_y', y_train)
    np.save('./nparrays/tnt/test_gray_X', X_test)
    np.save('./nparrays/tnt/test_gray_y', y_test)


# Creating talking and no talking Flow arrays 
if SAVE_TALKING_FLOW:
    print("Nothing here")


# Creating writing and no writing Gray numpy arrays for all the videos
if SAVE_WRITING_GRAY:
    w_path  = './videos/writing_all/'
    nw_path = './videos/no_writing_all/'

    X_wrt, y_wrt       = videosdir_2_gray_np(w_path,1)
    X_nowrt, y_nowrt   = videosdir_2_gray_np(nw_path,0)
    X                  = np.concatenate((X_wrt, X_nowrt), axis=0)
    y                  = np.concatenate((y_wrt, y_nowrt), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=19)
    np.save('./nparrays/wnw/train_gray_X_all', X_train)
    np.save('./nparrays/wnw/train_gray_y_all', y_train)
    np.save('./nparrays/wnw/test_gray_X_all', X_test)
    np.save('./nparrays/wnw/test_gray_y_all', y_test)


# Creating writing and no writing flow numpy arrays for all videos
if SAVE_WRITING_FLOW:
    num_samples = 30
    w_path  = './videos/writing_50x50/'
    nw_path = './videos/no_writing_50x50/'
    X_wrt, y_wrt       = videosdir_2_flow_np(w_path,1,num_samples)
    X_nowrt, y_nowrt   = videosdir_2_flow_np(nw_path,0,num_samples)
    X                  = np.concatenate((X_wrt, X_nowrt), axis=0)
    y                  = np.concatenate((y_wrt, y_nowrt), axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=19)
    np.save('./nparrays/wnw/train_flow_X_' + str(num_samples), X_train)
    np.save('./nparrays/wnw/train_flow_y_' + str(num_samples), y_train)
    np.save('./nparrays/wnw/test_flow_X_'  + str(num_samples),  X_test)
    np.save('./nparrays/wnw/test_flow_y_'  + str(num_samples),  y_test)
