"""
@author: Michele Donini
@email: mdonini@math.unipd.it

Toy test of the algorithm komd.py.
"""

# Test:
import sys
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from komd import KOMD
from cvxopt import matrix
import numpy as np

import matplotlib.pyplot as plt

# Binary classification problem
random_state = np.random.RandomState(0)
X, Y =  make_classification(n_samples=1000,
                            n_features=2,
                            n_informative=2,
                            n_redundant=0,
                            n_repeated=0,
                            n_classes=2,
                            n_clusters_per_class=2,
                            weights=None,
                            flip_y=0.0,
                            class_sep=1.0,
                            hypercube=True,
                            shift=0.0,
                            scale=1.0,
                            shuffle=True,
                            random_state=random_state)

X = matrix(X)
Y = matrix([1.0 if y>0 else -1.0 for y in Y])


# Train & Test:
pertr = 90
idtrain = range(0,len(Y) * pertr / 100)
idtest = range(len(Y) * pertr / 100,len(Y))
Ytr = Y[idtrain]
Yte = Y[idtest]

# Settings
ktype = 'rbf'   # type of kernel
gamma = 10.0**-1 # RBF parameter
l = 0.1 # lambda of KOMD


# KOMD
classifier = KOMD(lam=l, Kf = ktype, rbf_gamma = gamma)
y_score = classifier.fit(X[idtrain,:], Ytr).decision_function(X[idtest,:])
print 'AUC test:',roc_auc_score(np.array(Yte), np.array(y_score))



# Images, only if the X.size[1]==2 (2 dimensional datasets):
PLOT_THE_CLASS = True
if PLOT_THE_CLASS and X.size[1] == 2:
    ranktestnorm = [ (2 * (r - np.min(y_score))) / (np.max(y_score) - np.min(y_score)) - 1.0 for r in y_score]
    plt.figure(1)
    plt.scatter(X[idtrain, 0], X[idtrain, 1], marker='*', s = 140, c=Ytr, cmap='spring')
    plt.scatter(X[idtest, 0], X[idtest, 1], marker='o', s = 180, c=ranktestnorm, cmap='spring')
    plt.colorbar()
    plt.show()
