import os

import numpy as np
from sklearn.datasets import make_blobs, make_moons

from basic_perceptron import Perceptron
from best_perceptron import PerceptronBest
from util import visualize


def basic_clusters():
    x_arr, true_labels = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5]])
    perc = Perceptron()
    perc.fit(x_arr, true_labels)
    visualize(x_arr, true_labels, np.array(
        perc.predict(x_arr)), perc.w, "../data/clusters.png")


def basic_moons():
    x_arr, true_labels = make_moons(400, noise=0.075)
    perc = Perceptron()
    perc.fit(x_arr, true_labels)
    visualize(x_arr, true_labels, np.array(
        perc.predict(x_arr)), perc.w, "../data/moons.png")


def best_clusters():
    x_arr, true_labels = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5]])
    perc = PerceptronBest()
    perc.fit(x_arr, true_labels)
    visualize(x_arr, true_labels, np.array(perc.predict(x_arr)),
              perc.w, "../data/best_clusters.png")


def best_moons():
    x_arr, true_labels = make_moons(400, noise=0.075)
    perc = PerceptronBest()
    perc.fit(x_arr, true_labels)
    visualize(x_arr, true_labels, np.array(perc.predict(x_arr)),
              perc.w, "../data/best_moons.png")


if __name__ == '__main__':
    if not os.path.exists("../data"):
        os.mkdir("../data")
    basic_clusters()
    basic_moons()
    best_clusters()
    best_moons()
