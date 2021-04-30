import numpy as np

class MyNaturalCubicSpline:
    def fit(self, X_train, y_train, k):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.k = np.array(k)
        X_train_new = np.column_stack((np.ones(len(X_train)), X_train))
        d_Km1 = (np.where(np.power(X_train-k[-2], 3) < 0, 0, np.power(X_train-k[-2], 3))-np.where(np.power(X_train-k[-1], 3) < 0, 0, np.power(X_train-k[-1], 3)))/(k[-2]-k[-1])
        for i in range(len(k)-2):
            d = (np.where(np.power(X_train-k[i], 3) < 0, 0, np.power(X_train-k[i], 3))-np.where(np.power(X_train-k[-1], 3) < 0, 0, np.power(X_train-k[-1], 3)))/(k[i]-k[-1])
            X_train_new = np.column_stack((X_train_new, d-d_Km1))
        self.beta = np.linalg.inv(np.transpose(X_train_new).dot(X_train_new)).dot(np.transpose(X_train_new)).dot(y_train)
    def predict(self, X_test):
        X_test = np.array(X_test)
        X_test_new = np.column_stack((np.ones(len(X_test)), X_test))
        d_Km1 = (np.where(np.power(X_test-self.k[-2], 3) < 0, 0, np.power(X_test-self.k[-2], 3))-np.where(np.power(X_test-self.k[-1], 3) < 0, 0, np.power(X_test-self.k[-1], 3)))/(self.k[-2]-self.k[-1])
        for i in range(len(self.k)-2):
            d = (np.where(np.power(X_test-self.k[i], 3) < 0, 0, np.power(X_test-self.k[i], 3))-np.where(np.power(X_test-self.k[-1], 3) < 0, 0, np.power(X_test-self.k[-1], 3)))/(self.k[i]-self.k[-1])
            X_test_new = np.column_stack((X_test_new, d-d_Km1))
        return X_test_new.dot(self.beta)
