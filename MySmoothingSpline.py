import numpy as np

class MySmoothingSpline:
    def __init__(self, alpha=1):
        self.alpha = alpha
        
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train_new = np.column_stack((np.ones(len(self.X_train)), self.X_train))
        d_Km1 = (np.where(np.power(self.X_train-self.X_train[-2], 3) < 0, 0, np.power(self.X_train-self.X_train[-2], 3))-np.where(np.power(self.X_train-self.X_train[-1], 3) < 0, 0, np.power(self.X_train-self.X_train[-1], 3)))/(self.X_train[-2]-self.X_train[-1])
        for i in range(len(self.X_train)-2):
            d = (np.where(np.power(self.X_train-self.X_train[i], 3) < 0, 0, np.power(self.X_train-self.X_train[i], 3))-np.where(np.power(self.X_train-self.X_train[-1], 3) < 0, 0, np.power(self.X_train-self.X_train[-1], 3)))/(self.X_train[i]-self.X_train[-1])
            X_train_new = np.column_stack((X_train_new, d-d_Km1))
        delta = np.zeros((len(self.X_train)-2, len(self.X_train)))
        for i in range(len(self.X_train)-2):
            delta[i, i] = 1/(self.X_train[i+1]-self.X_train[i])
            delta[i, i+1] = -1/(self.X_train[i+1]-self.X_train[i])-1/(self.X_train[i+2]-self.X_train[i+1])
            delta[i, i+2] = 1/(self.X_train[i+2]-self.X_train[i+1])
        W = np.zeros((len(self.X_train)-2, len(self.X_train)-2))
        for i in range(1, len(self.X_train)-2):
            W[i-1, i] = W[i, i-1] = (self.X_train[i+1]-self.X_train[i])/6
            W[i, i] = (self.X_train[i+2]-self.X_train[i])/3
        omega = np.transpose(delta).dot(np.linalg.inv(W)).dot(delta)
        self.beta = np.linalg.inv(np.transpose(X_train_new).dot(X_train_new)+self.alpha*omega).dot(np.transpose(X_train_new)).dot(y_train)
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        X_test_new = np.column_stack((np.ones(len(X_test)), X_test))
        d_Km1 = (np.where(np.power(X_test-self.X_train[-2], 3) < 0, 0, np.power(X_test-self.X_train[-2], 3))-np.where(np.power(X_test-self.X_train[-1], 3) < 0, 0, np.power(X_test-self.X_train[-1], 3)))/(self.X_train[-2]-self.X_train[-1])
        for i in range(len(self.X_train)-2):
            d = (np.where(np.power(X_test-self.X_train[i], 3) < 0, 0, np.power(X_test-self.X_train[i], 3))-np.where(np.power(X_test-self.X_train[-1], 3) < 0, 0, np.power(X_test-self.X_train[-1], 3)))/(self.X_train[i]-self.X_train[-1])
            X_test_new = np.column_stack((X_test_new, d-d_Km1))
        return X_test_new.dot(self.beta)
