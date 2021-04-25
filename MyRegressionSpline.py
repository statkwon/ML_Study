import numpy as np

class MyRegressionSpline:
  def fit(self, X_train, y_train, m, k):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    self.m = m
    self.k = np.array(k)
    X_train_new = np.power(X_train, 0)
    for i in range(1, m+1):
      X_train_new = np.column_stack((X_train_new, np.power(X_train, i)))
    for i in range(len(k)):
      X_train_new = np.column_stack((X_train_new, np.where(np.power(X_train-k[i], m) < 0, 0, np.power(X_train-k[i], m))))
    self.beta = np.linalg.inv(np.transpose(X_train_new).dot(X_train_new)).dot(np.transpose(X_train_new)).dot(y_train)
  def predict(self, X_test):
    X_test = np.array(X_test)
    X_test_new = np.power(X_test, 0)
    for i in range(1, self.m+1):
      X_test_new = np.column_stack((X_test_new, np.power(X_test, i)))
    for i in range(len(self.k)):
      X_test_new = np.column_stack((X_test_new, np.where(np.power(X_test-self.k[i], self.m) < 0, 0, np.power(X_test-self.k[i], self.m))))
    return X_test_new.dot(self.beta)
