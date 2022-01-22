import numpy as np

class MyPerceptronLearningAlgorithm:
    def __init__(self, max_iter=100):
        self.max_iter = max_iter
    
    def fit(self, X_train, y_train, lr):
        ones = np.transpose(np.array([1]*len(X_train)))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        beta = np.array([0]*X_train.shape[1])
        beta0 = 0
        status = True ; tmp = 1
        while status:
            status = False
            yhat = np.array([-1 if i < 0 else 1 for i in (X_train.dot(beta) + ones.dot(beta0))])
            for i in range(len(X_train)):
                if y_train[i] != yhat[i]:
                    beta = beta + lr*y_train[i]*X_train[i]
                    beta0 = beta0 + lr*y_train[i]
                    tmp += 1
                    status = True
            if tmp > len(X_train)*self.max_iter:
                break
        self.beta_new = beta
        self.beta0_new = beta0
        
    def predict(self, X_test):
        ones = np.transpose(np.array([1]*len(X_test)))
        X_test = np.array(X_test)
        return np.array([-1 if i < 0 else 1 for i in (X_test.dot(self.beta_new) + self.beta0_new)])
