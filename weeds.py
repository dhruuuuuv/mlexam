import csv
import math
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py

import sklearn
from sklearn import linear_model, neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import zero_one_loss
import svms

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(float, data[i]))
            data[i] = point
        return data

def logreg():
    wtrain = import_dataset("ML2016WeedCropTrain.csv")
    wtest = import_dataset("ML2016WeedCropTest.csv")

    wtrain_whole = np.array(wtrain)
    wtest_whole = np.array(wtest)

    train_x = wtrain_whole[:,:-1]
    test_x = wtest_whole[:,:-1]

    train_y = wtrain_whole[:,-1]
    test_y = wtest_whole[:,-1]

# using co-ordinate descent
    logreg = linear_model.LogisticRegression(solver='liblinear')

    logreg.fit(train_x, train_y)

    params = logreg.coef_
    print("parameters of model: {}".format(params))

    bias = logreg.intercept_
    print("bias of model: {}".format(bias))

    y_pred_train = logreg.predict(train_x)
    y_pred_test = logreg.predict(test_x)

    z_one_loss_test = zero_one_loss(test_y, y_pred_test)
    z_one_loss_train = zero_one_loss(train_y, y_pred_train)

    print("01 loss test: {}".format(z_one_loss_test))
    print("01 loss train: {}".format(z_one_loss_train))

def perform_svm():
    wtrain = import_dataset("ML2016WeedCropTrain.csv")
    wtest = import_dataset("ML2016WeedCropTest.csv")

    wtrain_whole = np.array(wtrain)
    wtest_whole = np.array(wtest)

    train_x = wtrain_whole[:,:-1]
    test_x = wtest_whole[:,:-1]

    train_y = wtrain_whole[:,-1]
    test_y = wtest_whole[:,-1]

    j = jaakkola(train_x, train_y)
    gamma_j = 1 / (2 * (j**2))
    print(gamma_j)

    svms.svm(train_x, train_y, test_x, test_y)

def jaakkola(xs, ys):
    G = []
    for i, x in enumerate(xs):
        min = 100000
        # labx = ys[i]

        # x-indexes excluding current
        x_indices = np.arange(len(xs))
        x_indices = x_indices[np.arange(len(x_indices))!=i]

        for xind in x_indices:
            if ys[i] != ys[xind]:
                ed = np.linalg.norm(x - xs[xind])
                if ed <= min:
                    min = ed

        G.append(min)

    j = np.median(G)
    return j
