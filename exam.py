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

import galaxies

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(float, data[i]))
            data[i] = point
        return data

def main():
    redshiftss_train = "ML2016SpectroscopicRedshiftsTrain.dt"
    redshiftss_test = "ML2016SpectroscopicRedshiftsTest.dt"

    redshifte_train = "ML2016EstimatedRedshiftsTrain.dt"
    redshifte_test = "ML2016EstimatedRedshiftsTest.dt"

    galaxies_train = "ML2016GalaxiesTrain.dt"
    galaxies_test = "ML2016GalaxiesTest.dt"

    rs_ss_train = import_dataset(redshiftss_train)
    rs_ss_test = import_dataset(redshiftss_test)

    rs_e_train = import_dataset(redshifte_train)
    rs_e_test = import_dataset(redshifte_test)

    g_train = import_dataset(galaxies_train)
    g_test = import_dataset(galaxies_test)

    trainvar = galaxies.data_prep(rs_ss_train, rs_ss_test, rs_e_test)

    galaxies.linreg(g_train, g_test, rs_ss_train, rs_ss_test, trainvar)

    galaxies.knn(g_train, g_test, rs_ss_train, rs_ss_test)


if __name__ == '__main__':
    main()
