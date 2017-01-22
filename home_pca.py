import csv
import math
import operator
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py

def compute_eigenvalues(data):
        ds_nolab = data[:,:-1]

        # compute the mean
        meanmat = np.array(np.mean(ds_nolab, axis=1))
        covmat = np.cov(ds_nolab)

        # compute eigenvalues and vectors
        eigen_values, eigen_vectors = (np.linalg.eigh(covmat))
        # print(eigen_vectors.shape)
        # print(eigen_vectors)

        eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]

        # sort eigenpairs by decreasing eigenvalue
        eigen_pairs.sort(key= lambda x: x[0], reverse=True)

        return (meanmat, eigen_pairs)

def plot_eigenspectrum(eigen_values):
    fig = plt.figure()
    ax = plt.subplot(211)
    plt.grid(True)

    print(eigen_values)

    cumulative_eigen_values = np.cumsum(eigen_values)

    x = np.array(range(len(eigen_values)))

    log_eigen = []

    log_eigen = [math.log(x) for x in eigen_values]

    # for i in range(len(eigen_values)):
    #     log_eigen.append(math.log(eigen_values[i]))

    plt.xlabel("number of eigenvector")
    plt.ylabel("log of eigenvalue")
    plt.title("eigenspectrum of covariance matrix")
    plt.plot(x, log_eigen)
    # plt.plot(x, eigen_values)

    plt.xlim((min(x), max(x)))
    plt.ylim((min(log_eigen), max(log_eigen)))
    plt.show()

def get_label_vector(data):
    return data[:,-1]


def compute_number_vectors(eigen_pairs, varval):
    eigen_values = []

    for i in eigen_pairs:
        eigen_values.append(i[0])

    eigen_values = np.array(eigen_values)

    eigen_values = (eigen_values / sum(eigen_values))
    cumulative_eigen_values = np.cumsum(eigen_values)

    vectors_needed = 0
    for i in range(len(cumulative_eigen_values)):
        if cumulative_eigen_values[i] >= varval:
            vectors_needed = i
            break

    return vectors_needed

def plot_2_pc(vec1, vec2, label_vec):

    # print(vec1.shape)
    # print(vec2.shape)
    # print(label_vec.shape)

    fig = plt.figure()
    ax = plt.subplot(211)
    plt.grid(True)


    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    plt.title("Plot of first 2 Principal Components")
    # plt.scatter(vec1, vec2)

    zipped_tuples = list(zip(vec1, vec2))
    # print(zipped_tuples)

    for i in range(len(zipped_tuples)):
        color = get_color(label_vec[i])
        plt.scatter(zipped_tuples[i][0], zipped_tuples[i][1], color=color[0], label=color[1])

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=3, prop={'size':10})

    plt.show()

def plot_2_pc_cc(vec1, vec2, label_vec, ccs):

    fig = plt.figure()
    ax = plt.subplot(211)
    plt.grid(True)


    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    plt.title("Plot of first 2 Principal Components")
    # plt.scatter(vec1, vec2)

    zipped_tuples = list(zip(vec1, vec2))

    for i in range(len(zipped_tuples)):
        color = get_colour_from_label(label_vec[i])
        plt.scatter(zipped_tuples[i][0], zipped_tuples[i][1], color=color[0], label=color[1])

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc=3, prop={'size':10})

    plt.show()

def get_color(label):
    if label:
        return ['b', "crops"]
    else:
        return ['r', "weeds"]

def get_colour_from_label(label):
    colours = ['b', 'g', 'r', 'c', 'm']
    labels = ["round", "up triangle", "diamond", "down triangle", "octagon"]

    if (label >= 0 and label <= 10) or (label >= 15 and label <= 17) or (label >= 32 and label <= 42):
        return (colours[0], labels[0])

    elif (label == 11) or (label >= 18 and label <= 31):
        return (colours[1], labels[1])

    elif label == 12:
        return (colours[2], labels[2])

    elif label == 13:
        return (colours[3], labels[3])

    elif label == 14:
        return (colours[4], labels[4])

    else:
        print("invalid label")

def get_k_vectors(k, eigen_pairs):
    vectors = []
    for i in range(k):
        vectors.append(eigen_pairs[i][1])

    return np.array(vectors).T

def transform_data(data, pcs):
    mean = data.mean(axis=1)
    # minmean = []
    # for i in range(len(data)):
    #     minmean.append(data[i] - mean)

    norm_data = data - mean[:, np.newaxis]

    # print(norm_data)
    # print(data.shape)
    # print(pcs.shape)

    return pcs.T.dot(norm_data)

def pca(filearg, k, num, centroids):
    test = np.array([[1,1,1], [2, 2, 2], [3, 3, 3]])

    data = np.loadtxt(filearg, delimiter=',')
    wholeds = np.array(data)

    # print("data shape")
    # print(wholeds.shape)
    # if num != 0:
    #     print(centroids.shape)
    #     np.append(centroids,[[0, 0, 0, 0]], axis=1)
    #     np.append(wholeds, centroids, axis=0)
    #     print(wholeds)

    mean, eigen_pairs = compute_eigenvalues(wholeds)
    # print(eigen_pairs[1])

    # plot_eigenspectrum(eigen_pairs)
    vectors_needed = compute_number_vectors(eigen_pairs, 0.9)

    alleigenvec = get_k_vectors(len(eigen_pairs), eigen_pairs)

    k_pc = get_k_vectors(vectors_needed, eigen_pairs)
    print("number of vectors needed: " + repr(vectors_needed))

    lv = get_label_vector(wholeds)

    plot_2_pc(k_pc[:,0], k_pc[:,1], lv)
    # print("v1 shape")
    # print(k_pc[0].shape)

    ds_nolab = data[:,:-1]

    # projected_data = transform_data(ds_nolab, k_pc)

    return (mean, k_pc, alleigenvec, lv)
