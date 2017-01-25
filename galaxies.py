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
    params = {"n_neighbors": np.arange(1, 31, 2)}
    model = neighbors.KNeighborsRegressor(n_jobs=-1)
    grid = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=5)
    grid.fit(gtrain, labeltrain)
    acc = grid.score(gtest, labeltest)
    acc2 = grid.score(gtrain, labeltrain)

    print("grid search mse test: {}".format(-acc))
    print("grid search mse train: {}".format(-acc2))
    print("grid search best param: {}".format(grid.best_params_))

    # k = 3
    # knn = neighbors.KNeighborsRegressor(3)
    # y_ = knn.fit(gtrain, labeltrain).predict(gtest)
    #
    # # print(y_)
    # mse = ((y_ - labeltest) ** 2).mean(axis=None)
    # print("mse on knn, k = %.1f: " % k + repr(mse))


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
