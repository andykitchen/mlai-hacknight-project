import os
import sys
from os import listdir
from os.path import isfile, isdir, join, splitext
from scipy import misc
import numpy as np

def get_array_from_img(path):
    """ Take a .jpg file and return it's pixel data in an array. """
    face = misc.imread(path)
    return face

def extract(verbose=False):
    """ Return facial image data from folder. """
    data, labels = [], []
    nloc_in_project, num_problems, num_questions = 0,0,0
    search_dir = os.getcwd() # we want the parent directory
    if verbose: print(search_dir)
    for folder, subfolders, files in os.walk(search_dir, topdown=False):
        if 'data' in folder:
            for f in [f for f in files if f.endswith(".jpg") and not f.startswith(".")]:
                arr = get_array_from_img(os.path.join(folder, f))
                data.append(arr)
                labels.append(int(folder[-2:]))
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels # shape -> (300, 213, 320, 3)


def load_data(one_hot=True):
    """ Take facial image data and structure it into training and test datasets. """
    X, Y  = extract()
    if one_hot:
        n_labels = np.amax(Y) # max label is number of labels
        Y = np.equal.outer(Y, np.arange(n_labels+1)).astype(np.float)
        Y = Y[:,1:] # cut off 0 label
    # Split data into train and test
    mask = np.random.rand(len(X)) < 0.8
    testX = X[mask]
    X = X[~mask] # inverse of mask for training data
    testY = Y[mask]
    Y = Y[~mask] # inverse mask again for training labels

    return X, Y, testX, testY
