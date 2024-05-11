import cvxopt
from cvxopt import matrix
import numpy as np


class Polynomial:
    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        if A.ndim == 1 and B.ndim == 1:
            K = A.dot(B)
        elif A.ndim == 2 and B.ndim == 2:
            K = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2:  np.dot(x1, x2), 1, B), 1, A)
        elif A.ndim == 1 and B.ndim == 2:
            K = np.apply_along_axis(lambda x1: np.dot(x1, A), 1, B)
        elif A.ndim == 2 and B.ndim == 1:
            K = np.apply_along_axis(lambda x1: np.dot(x1, B), 1, A)

        return (1 + K) ** self.M


class RBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        if A.ndim == 1 and B.ndim == 1:
            K = np.linalg.norm(A - B)
        elif A.ndim == 2 and B.ndim == 2:
            K = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2: np.linalg.norm(x1 - x2), 1, B), 1, A)
        elif A.ndim == 1 and B.ndim == 2:
            K = np.apply_along_axis(lambda x1: np.linalg.norm(A - x1), 1, B)
        elif A.ndim == 2 and B.ndim == 1:
            K = np.apply_along_axis(lambda x1: np.linalg.norm(x1 - B), 1, A)

        return np.exp(-K**2 / (2 * self.sigma**2))


class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_):
        self.l = lambda_
        self.kernel = kernel


    def fit(self, X, y):
        self.X = X

        K = self.kernel(X, X)
        self.a = np.linalg.inv((K + self.l * np.identity(K.shape[0]))).dot(y)

        return self


    def predict(self, X):
        X = self.kernel(self.X, X)
        return np.sum((self.a[np.newaxis].T * X), axis=0)


class SVR:
    def __init__(self, kernel, lambda_, epsilon):
        self.l = lambda_
        self.e = epsilon
        self.kernel = kernel
        self.C = 1. / lambda_


    def fit(self, X, y):
        self.X = X

        K = self.kernel(X, X)
        
        I = np.full((2, 2), 1) + np.fliplr(np.identity(2) * -2.)
        I = np.tile(I, K.shape)

        P = np.repeat(np.repeat(K,2, axis=0), 2, axis=1) * I

        q = np.repeat(y, 2).astype(np.float64)
        q[1::2] *= -1.
        q -= self.e
        q *= (-1)
        
        A = np.zeros(X.shape[0]*2)
        A[::2] = 1.
        A[1::2] = -1.
        A = A[np.newaxis, :]
        b = np.zeros(1)

        G = np.concatenate((np.identity(X.shape[0]*2), np.identity(X.shape[0]*2) * (-1.)), axis=0)
        h = np.concatenate((np.full(X.shape[0]*2, self.C), np.zeros(X.shape[0]*2)), axis=None)
        
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
        self.alpha = np.array(solution['x'])
        self.alpha = np.reshape(self.alpha, (self.X.shape[0], 2))
        self.alpha[np.arange(len(self.alpha)), self.alpha.argmin(axis=1)] = 0

        self.beta = solution['y'][0]

        return self


    def get_alpha(self):
        return self.alpha


    def get_b(self):
        return self.beta


    def predict(self, X):
        a = np.subtract(self.alpha[:, 0], self.alpha[:, 1])[np.newaxis].T
        X = self.kernel(self.X, X)
        return np.sum((a * X), axis=0) + self.beta