import numpy as np
from sklearn.datasets import make_blobs, make_moons

from scripts.basic_perceptron import Perceptron
from scripts.best_perceptron import PerceptronBest
from scripts.util import visualize


def basic_clusters():
    X, true_labels = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5]])
    c = Perceptron()
    c.fit(X, true_labels)
    visualize(X, true_labels, np.array(c.predict(X)), c.w)


def basic_moons():
    X, true_labels = make_moons(400, noise=0.075)
    c = Perceptron()
    c.fit(X, true_labels)
    visualize(X, true_labels, np.array(c.predict(X)), c.w)


def best_clusters():
    X, true_labels = make_blobs(400, 2, centers=[[0, 0], [2.5, 2.5]])
    c = PerceptronBest()
    c.fit(X, true_labels)
    visualize(X, true_labels, np.array(c.predict(X)), c.w)


def best_moons():
    X, true_labels = make_moons(400, noise=0.075)
    c = PerceptronBest()
    c.fit(X, true_labels)
    visualize(X, true_labels, np.array(c.predict(X)), c.w)


if __name__ == '__main__':
    basic_clusters()
    basic_moons()
    best_clusters()
    best_moons()
