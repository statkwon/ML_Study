import math
import numpy as np

class MyNWKernelRegression:
    def __init__(self, kernel='Epanechnikov', width=10):
        self.kernel = kernel
        self.width = width
        
    def epanechnikov(self, x):
        return np.where(abs(x) <= 1, 0.75*(1-x**2), 0)
    
    def tricube(self, x):
        return np.where(abs(x) <= 1, (1-abs(x)**3)**3, 0)
    
    def gaussian(self, x):
        return 1/np.sqrt(2*math.pi)*np.exp(-0.5*(x**2))
    
    def uniform(self, x):
        return np.where(abs(x) <= 1, 0.5, 0)
    
    def predict(self, X_train, y_train, X_test):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_pred = np.array([])
        for i in range(len(X_test)):
            if self.kernel == 'KNN':
                t = abs(X_train-X_test[i])/abs(X_train-X_test[i])[np.argsort(abs(X_train-X_test[i]))==self.width][0]
                d = self.uniform(t)
            else:
                t = abs(X_train-X_test[i])/self.width
                if self.kernel == 'Epanechnikov':
                    d = self.epanechnikov(t)
                elif self.kernel == 'Tri-Cube':
                    d = self.tricube(t)
                else:
                    d = self.gaussian(t)
            y_pred = np.append(y_pred, np.sum(d*y_train)/np.sum(d))
        return y_pred
