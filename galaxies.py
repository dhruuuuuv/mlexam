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

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(float, data[i]))
            data[i] = point
        return data

def data_prep(train, stest, etest):
    train = np.array(train)
    stest = np.array(stest)
    etest = np.array(etest)

    var = np.var(train)
    print("variance redsstrain: " + repr(var))

    mse = ((etest - stest) ** 2).mean(axis=None)
    print("mse on rsss: " + repr(mse))

    return var

def knn(gtrain, gtest, labeltrain, labeltest):


def linreg(gtrain, gtest, labeltrain, labeltest, var):
    gtrain = np.array(gtrain)
    gtest = np.array(gtest)
    labeltrain = np.array(labeltrain)
    labeltest = np.array(labeltest)

    # make linear model
    regr = linear_model.LinearRegression()

    # train using galaxies
    regr.fit(gtrain, labeltrain)
    print('coefficients linreg: \n', regr.coef_)

    # mse
    mse_test = np.mean((regr.predict(gtest) - labeltest) ** 2)
    print("MSE linreg test: %.8f" % mse_test)
    # mse
    mse_train = np.mean((regr.predict(gtrain) - labeltrain) ** 2)
    print("MSE linreg train: %.8f" % mse_train)

    print("normalised MSE test: %.8f" % (mse_test/var))

    print("normalised MSE train: %.8f" % (mse_train/var))

    # linreg plot
    # plt.scatter(gtest, labeltest, color = 'black')
    # plt.plot(gtest, regr.predict(gtest), color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
