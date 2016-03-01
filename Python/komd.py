"""
@author: Michele Donini
@email: mdonini@math.unipd.it

KOMD is a kernel machine for ranking.

A Kernel Method for the Optimization of the Margin Distribution
by F. Aiolli, G. Da San Martino, and A. Sperduti.
"""
 

from cvxopt import matrix, solvers
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel


class KOMD(BaseEstimator, ClassifierMixin):
    """KOMD"""
    def __init__(self, lam = 0.1, Kf = None, rbf_gamma = 0.1, poly_deg = 2.0, poly_coeff = 0.0, verbose = False):
        self.lam = lam
        self.verbose = verbose
        
        self.gamma = None
        self.bias = None
        self.X = None
        self.labels = None
        
        self.rbf_gamma = rbf_gamma
        self.poly_deg = poly_deg
        self.poly_coeff = poly_coeff
        
        self.Kf = Kf

    def __kernel_definition__(self):
        if self.Kf == 'rbf':
            return lambda X,Y : rbf_kernel(X,Y,self.rbf_gamma)
        if self.Kf == 'poly':
            return lambda X,Y : polynomial_kernel(X, Y, degree=self.poly_deg, gamma=None, coef0=self.poly_coeff)
        if self.Kf == None or self.Kf == 'linear':
            return lambda X,Y : linear_kernel(X,Y)
    
    def fit(self, X, labels):
        ''' 
            X : matrix of the examples
            labels : array of the labels
        '''
        self.X = X
        self.labels = labels
        
        Kf = self.__kernel_definition__()
        YY = matrix(np.diag(list(matrix(labels))))
        #YY = matrix(np.diag(labels)) #questa YY errata!
        ker_matrix = matrix(Kf(X,X))
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam]*len(labels)))
        Q = 2*(KLL+LID)
        p = matrix([0.0]*len(labels))
        G = -matrix(np.diag([1.0]*len(labels)))
        h = matrix([0.0]*len(labels),(len(labels),1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in labels],[1.0 if lab2==-1 else 0 for lab2 in labels]]).T
        b = matrix([[1.0],[1.0]],(2,1))
        
        solvers.options['show_progress']=False#True
        sol = solvers.qp(Q,p,G,h,A,b)
        self.gamma = sol['x']     
        
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias
        return self
        
    def predict(self, X):
        return np.array([1 if p >=0 else -1 for p in self.decision_function(X)])

    def get_params(self, deep=True):
        # this estimator has parameters:
        return {"lam": self.lam, "Kf": self.Kf, "rbf_gamma":self.rbf_gamma,
                "poly_deg":self.poly_deg, "poly_coeff":self.poly_coeff}

    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter,value)
        return self

    
    def decision_function(self, X):
        ''' Distance of the samples X to the separating hyperplane.
            Parameters:	
                X : array-like, shape = [n_samples, n_features]
            Returns:	
                X : array-like, shape = [n_samples, 1]
                Returns the decision function of the sample.
        '''
        Kf = self.__kernel_definition__()
        #YY = matrix(np.diag(self.labels)) # YY errata!
        YY = matrix(np.diag(list(matrix(self.labels))))
        ker_matrix = matrix(Kf(X,self.X))
        z = ker_matrix*YY*self.gamma
        return z-self.bias
    
    def predict_proba(self, X):
        return self.decision_function(X)

            



