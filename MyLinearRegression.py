import numpy as np

class MyLinearRegerssion:
    def fit(self, X_train, y_train):
        ones = np.ones(len(X_train))
        X_train = np.array(X_train)
        X_train = np.column_stack((np.ones(len(X_train)), X_train))
        y_train = np.array(y_train)
        self.beta = np.linalg.inv(np.transpose(X_train).dot(X_train)).dot(np.transpose(X_train)).dot(y_train)
        
    def predict(self, X_test):
        ones = np.ones(len(X_test))
        X_test = np.array(X_test)
        X_test = np.column_stack((np.ones(len(X_test)), X_test))
        return X_test.dot(self.beta)
