import csv
import math
import operator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py

import sklearn
from sklearn import linear_model, neighbors, decomposition
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import zero_one_loss
from sklearn.cluster import KMeans
import svms, home_pca

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(float, data[i]))
            data[i] = point
        return data

def split_data():
    wtrain = import_dataset("ML2016WeedCropTrain.csv")
    wtest = import_dataset("ML2016WeedCropTest.csv")

    wtrain_whole = np.array(wtrain)
    wtest_whole = np.array(wtest)

    train_x = wtrain_whole[:,:-1]
    test_x = wtest_whole[:,:-1]

    train_y = wtrain_whole[:,-1]
    test_y = wtest_whole[:,-1]

    return (train_x, train_y, test_x, test_y)



def logreg(train_x, train_y, test_x, test_y):

# using co-ordinate descent
    logreg = linear_model.LogisticRegression(solver='liblinear')

    logreg.fit(train_x, train_y)

    params = logreg.coef_
    print("parameters of model: {}".format(params))

    bias = logreg.intercept_
    print("bias of model: {}".format(bias))

    y_pred_train = logreg.predict(train_x)
    y_pred_test = logreg.predict(test_x)

    z_one_loss_test = 1 - zero_one_loss(test_y, y_pred_test)
    z_one_loss_train = 1 - zero_one_loss(train_y, y_pred_train)

    print("01 loss test: {}".format(z_one_loss_test))
    print("01 loss train: {}".format(z_one_loss_train))

def normalised_methods():
    wtrain = import_dataset("ML2016WeedCropTrain.csv")
    wtest = import_dataset("ML2016WeedCropTest.csv")

    wtrain_whole = np.array(wtrain)
    wtest_whole = np.array(wtest)

    # -- DELETE WHEN DONE --

    np.set_printoptions(suppress=True)

    # print("RANGES")
    # print(svms.calc_range(wtrain_whole))

    train_x, train_y, test_x, test_y = svms.norm(wtrain_whole, wtest_whole)

    # print("RANGES NORM")
    # print(svms.calc_range(train_x))

    print("normalised logistic regression")
    logreg(train_x, train_y, test_x, test_y)

    print("normalised SVMs")
    perform_svm(train_x, train_y, test_x, test_y)

def clustering(train_x, train_y, test_x, test_y, center_init):
    kmeans = KMeans(n_clusters=2, n_init=1, max_iter=1000, init=center_init, algorithm="full", n_jobs=-1).fit(train_x)
    # kmeans = KMeans(n_clusters=2, init=center_init, n_jobs=-1).fit(train_x)
    # kmeans = KMeans(n_clusters=2, n_jobs=-1).fit(train_x)

    # comparison = (kmeans.labels_ == train_y)
    # true_values = [x for x in comparison if x == True]
    # false_values = [x for x in comparison if x == False]
    # print(len(true_values)/len(train_y))
    # print(len(false_values)/len(train_y))
    # print(kmeans.cluster_centers_)
    return kmeans


def principal_ca(train_x, train_y, test_x, test_y):
    # print(train_x.shape)
    np.set_printoptions(suppress=True)


    pca = decomposition.PCA()
    pca.fit(train_x)
    # print(pca.n_components_)

    pcs = pca.components_
    # print(pca.components_.shape)
    # print(pca.components_)
    sum_val = 0

    v1 = np.dot(train_x, pca.components_[0])
    v2 = np.dot(train_x, pca.components_[1])

    for i, x in enumerate(pca.explained_variance_ratio_):
        sum_val += pca.explained_variance_ratio_[i]
        # print(sum_val)

    eigvalues = pca.explained_variance_ratio_

    # home_pca.plot_eigenspectrum(eigvalues)

    # pca2 = decomposition.PCA(0.90)
    # pca2.fit(train_x)
    # print(pca2.n_components_)

    # home_pca.plot_2_pc(v1, v2, train_y)

    return (pca.components_, v1, v2)


def perform_svm(train_x, train_y, test_x, test_y):

    j = jaakkola(train_x, train_y)
    gamma_j = 1 / (2 * (j**2))

    print("jaakkola sigma: {}".format(j))
    print("jaakkola gamma: {}".format(gamma_j))

    svms.svm(train_x, train_y, test_x, test_y, gamma_j)

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
