import numpy as np

class MyLogisticRegression:
    def __init__(self, max_iter=10):
        self.max_iter = max_iter
    def fit(self, X_train, y_train):
        ones = np.transpose(np.array([[1]*len(X_train)]))
        X_train = np.concatenate((ones, np.array(X_train)), axis=1)
        y_train = np.array(y_train)
        beta = np.array([0]*X_train.shape[1])
        for i in range(self.max_iter):
            p1 = np.exp(X_train.dot(beta))/(1+np.exp(X_train.dot(beta)))
            p0 = 1-p1
            W = np.diag(p0*p1)
            beta = beta + np.linalg.inv(np.transpose(X_train).dot(W).dot(X_train)).dot(np.transpose(X_train)).dot(y_train-p1)
            self.beta_new = beta
    def predict(self, X_test):
        ones = np.transpose(np.array([[1]*len(X_test)]))
        X_test = np.concatenate((ones, np.array(X_test)), axis=1)
        return (np.exp(X_test.dot(self.beta_new))/(1+np.exp(X_test.dot(self.beta_new))) > 0.5).astype('int')
