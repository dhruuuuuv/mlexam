import csv
import math
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py

import sklearn
from sklearn import linear_model

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=' ')
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

def linreg(gtrain, gtest, rssstrain, rssstest):
