import numpy as np

class MyRidgeRegerssion:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    def fit(self, X_train, y_train):
        ones = np.ones(len(X_train))
        X_train = np.array(X_train)
        X_train = np.column_stack((np.ones(len(X_train)), X_train))
        y_train = np.array(y_train)
        self.beta = np.linalg.inv(np.transpose(X_train).dot(X_train)+self.alpha*np.identity(X_train.shape[1])).dot(np.transpose(X_train)).dot(y_train)
        
    def predict(self, X_test):
        ones = np.ones(len(X_test))
        X_test = np.array(X_test)
        X_test = np.column_stack((np.ones(len(X_test)), X_test))
        return X_test.dot(self.beta)
