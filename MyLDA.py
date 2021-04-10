import numpy as np

class MyLDA:
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        self.pi = [len(X_train[y_train==i])/len(X_train) for i in np.unique(y_train)]
        self.mu = [np.sum(X_train[y_train==i], axis=0)/len(X_train[y_train==i]) for i in np.unique(y_train)]
        self.sigma = np.sum([(np.transpose(X_train[y_train==i]-self.mu[i])).dot(X_train[y_train==i]-self.mu[i]) for i in np.unique(y_train)], axis=0)/(len(X_train)-len(np.unique(y_train)))
    
    def predict(self, X_test):
        delta = [X_test.dot(np.linalg.inv(self.sigma)).dot(self.mu[i])-0.5*self.mu[i].dot(np.linalg.inv(self.sigma)).dot(self.mu[i])+np.log(self.pi[i]) for i in np.unique(y)]
        yhat = np.argmax(delta, axis=0)
        return yhat
