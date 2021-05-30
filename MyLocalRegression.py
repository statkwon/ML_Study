import math
import numpy as np

class MyLocalRegression:
    def __init__(self, kernel='Tri-Cube', width=10):
        self.kernel = kernel
        self.width = width
    
    def tricube(self, x):
        return np.where(abs(x) <= 1, (1-abs(x)**3)**3, 0)
    
    def epanechnikov(self, x):
        return np.where(abs(x) <= 1, 0.75*(1-x**2), 0)
    
    def gaussian(self, x):
        return 1/np.sqrt(2*math.pi)*np.exp(-0.5*(x**2))
    
    def predict(self, X_train, y_train, X_test):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_pred = np.array([])
        for i in range(len(X_test)):
            t = abs(X_train-X_test[i])/self.width
            if self.kernel == 'Tri-Cube':
                d = self.tricube(t)
            elif self.kernel == 'Epanechnikov':
                d = self.epanechnikov(t)
            else:
                d = self.gaussian(t)
            X_train_nonzero = np.column_stack((np.power(X_train[d != 0], 0), X_train[d != 0]))
            y_train_nonzero = y_train[d != 0]
            W = np.diag(d[d != 0])
            y_pred = np.append(y_pred, np.transpose(np.array((1, X_test[i]))).dot(np.linalg.inv(np.transpose(X_train_nonzero).dot(W).dot(X_train_nonzero))).dot(np.transpose(X_train_nonzero)).dot(W).dot(y_train_nonzero))
        return y_pred
