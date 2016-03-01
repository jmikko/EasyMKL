"""
@author: Michele Donini
@email: mdonini@math.unipd.it

Toy test of the algorithm EasyMKL.py.
"""

# Test:
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from EasyMKL import EasyMKL
from komd import KOMD
from cvxopt import matrix
import numpy as np

import matplotlib.pyplot as plt

# Binary classification problem
random_state = np.random.RandomState(0)
X, Y =  make_classification(n_samples=1000,
                            n_features=50,
                            n_informative=10,
                            n_redundant=10,
                            n_repeated=10,
                            n_classes=2,
                            n_clusters_per_class=5,
                            weights=None,
                            flip_y=0.0,
                            class_sep=0.5,
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

# Selected features for each weak kernel:
featlist = [[random_state.randint(0,X.size[1]) for i in range(5)] for j in range(50)]

# Generation of the weak Kernels:
klist = [rbf_kernel(X[:,f], gamma = 0.1) for f in featlist]
klisttr = [matrix(k)[idtrain,idtrain] for k in klist]
klistte = [matrix(k)[idtest,idtrain] for k in klist] 

# EasyMKL initialization:
l = 0.5 # lambda
easy = EasyMKL(lam=l, tracenorm = True)
easy.train(klisttr,Ytr)

# Evaluation:
rtr = roc_auc_score(np.array(Ytr),np.array(easy.rank(klisttr)))
print 'AUC EasyMKL train:',rtr
ranktest = np.array(easy.rank(klistte))
rte = roc_auc_score(np.array(Yte),ranktest)
print 'AUC EasyMKL test:',rte
print 'weights of kernels:', easy.weights


# Comparison with respect the single kernels:
print '\n\n\n\n\nSingle kernel analisys using KOMD:'
YYtr = matrix(np.diag(list(Ytr)))
for idx,f in enumerate(featlist):
    classifier = KOMD(lam=l, Kf = 'rbf', rbf_gamma = 0.1)
    y_score = classifier.fit(X[idtrain,f], Ytr).decision_function(X[idtest,f])
    print 'K with features:',f,'AUC test:',roc_auc_score(np.array(Yte), np.array(y_score))
    print '\t\t margin train: \t\t',(easy.gamma.T * YYtr * matrix(klist[idx])[idtrain,idtrain] * YYtr * easy.gamma)[0]
    print '\t\t weight assigned: \t',easy.weights[idx]


# Some (not so useful) images, only if the X.size[1]==2 (2 dimensional datasets):
PLOT_THE_CLASS = True
ranktestnorm = [ (2 * (r - np.min(ranktest))) / (np.max(ranktest) - np.min(ranktest)) - 1.0 for r in ranktest]
if PLOT_THE_CLASS and X.size[1] == 2:
    plt.figure(1)
    plt.scatter(X[idtrain, 0], X[idtrain, 1], marker='*', s = 140, c=Ytr, cmap='spring')
    plt.scatter(X[idtest, 0], X[idtest, 1], marker='o', s = 180, c=ranktestnorm, cmap='spring')
    plt.colorbar()
