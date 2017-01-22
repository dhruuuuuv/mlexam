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

import galaxies, weeds, home_pca

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(float, data[i]))
            data[i] = point
        return data

def part1():
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

        print("-- part 1.1 ---")
        trainvar = galaxies.data_prep(rs_ss_train, rs_ss_test, rs_e_test)

        print("-- part 1.2 ---")
        galaxies.linreg(g_train, g_test, rs_ss_train, rs_ss_test, trainvar)

        print("-- part 1.3 ---")
        galaxies.knn(g_train, g_test, rs_ss_train, rs_ss_test)

def part2():
    train_x, train_y, test_x, test_y = weeds.split_data()

    # print("-- part 2.1 --")
    # weeds.logreg(train_x, train_y, test_x, test_y)

    # print("-- part 2.2 --")
    # weeds.perform_svm(train_x, train_y, test_x, test_y)

    # print("-- part 2.3 --")
    # weeds.normalised_methods()

    # print("-- part 2.4 --")
    pcs, v1, v2 = weeds.principal_ca(train_x, train_y, test_x, test_y)
    pc_2 = pcs[:2, :]

    # print("-- part 2.5 --")
    centers = train_x[:2,:]
    kmeans = weeds.clustering(train_x, train_y, test_x, test_y, centers)
    final_centroids = kmeans.cluster_centers_
    ccv1 = np.dot(final_centroids, pcs[0])
    ccv2 = np.dot(final_centroids, pcs[1])
    print(final_centroids.shape)
    print(pcs[0].shape)
    print(pcs[1].shape)
    # print(ccv1)
    # print(ccv2)
    home_pca.plot_2_pc_cc(v1, v2, train_y, ccv1, ccv2)

def main():
    # part1()
    part2()

if __name__ == '__main__':
    main()
