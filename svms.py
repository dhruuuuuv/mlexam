import csv
import math
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py
import sklearn

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

def svm(data, labels, test, testlab, j):
    b = 10
    cvals = [b**-3, b**-2, b**-1, 1, b, b**2, b**3, b**4, b**5, b**6]
    gamma = [b**-3, b**-2, b**-1, 1, b, b**2, b**3]
    gammavals = [(i * j) for i in gamma]

    param_grid = [
    {'C' : cvals, 'gamma' : gammavals, 'kernel': ['rbf']}
    ]

    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring='accuracy')

    clf.fit(data, labels)

    print("best parameters: {}".format(clf.best_params_))

    print("accuracy on training set: {}".format(clf.score(data, labels)))
    print("accuracy on test set: {}".format(clf.score(test, testlab)))

def norm(train, test):

    tr = np.array(train)
    te = np.array(test)

    trl = tr[:,-1]
    tel = te[:,-1]

    trnl = tr[:,:-1]
    tenl = te[:,:-1]

    # print(te)
    # print(tenl)

    mean = np.mean(tr, axis=0)
    var = np.var(tr, axis=0)

    # print(mean, var)

    trn = preprocessing.scale(tr)
    ten = preprocessing.scale(te)

    trnl = preprocessing.scale(trnl)
    tenl = preprocessing.scale(tenl)

    for i, row in enumerate(trn):
        row[-1] = trl[i]

    for i, row in enumerate(ten):
        row[-1] = tel[i]

    meantest = np.mean(ten, axis=0)
    vartest = np.var(ten, axis=0)

    return (trnl, trl, tenl, tel)
