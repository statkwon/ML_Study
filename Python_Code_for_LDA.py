import warnings
import numpy as np

def _class_means(X, y):
    classes = np.unique(y)
    means = np.array([np.mean(X[y == i], axis=0) for i in classes])
    return means

def _class_cov(X, y, priors):
    _, p = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    cov = np.zeros((p, p))
    cov_ = np.array([np.cov(X[y == i], rowvar=False) for i in classes])
    for i in range(n_classes):
        cov += priors[i] * cov_[i]
    return cov

def softmax(X):
    X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X

class LinearDiscriminantAnalysis:
    """My Linear Discriminant Analysis"""
    
    def __init__(self, priors=None):
        self.priors = priors
    
    def _solve(self, X, y):
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(X, y, self.priors_)
        self.coef_ = self.means_.dot(np.linalg.inv(self.covariance_))
        self.intercept_ = -0.5 * np.diag(self.means_.dot(self.coef_.T)) + np.log(self.priors_)
    
    def fit(self, X, y):
        X = np.array(X) ; y = np.array(y)
        self.classes_ = np.unique(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)
        
        if n_samples == n_classes:
            raise ValueError('데이터의 개수는 범주의 개수보다 많아야합니다.')
        
        if self.priors is None:
            self.priors_ = np.array([sum(y == i) / len(y) for i in self.classes_])
        else:
            self.priors_ = np.array(self.priors)
        
        if any(self.priors_ < 0):
            raise ValueError('사전 확률은 0보다 커야합니다.')
        if not np.isclose(sum(self.priors_), 1):
            warnings.warn('사전 확률의 합이 1이 아닙니다. 값을 재조정합니다', UserWarning)
            self.priors_ = self.priors_ / sum(self.priors_)
        
        self._solve(X, y)
        
        return self
    
    def decision_function(self, X):
        X = np.array(X)
        scores = X.dot(self.coef_.T) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores
    
    def predict(self, X):
        decision = self.decision_function(X)
        y_pred = self.classes_.take(decision.argmax(1))
        return y_pred
    
    def predict_proba(self, X):
        decision = self.decision_function(X)
        return softmax(decision)
    
    def predict_log_proba(self, X):
        prediction = self.predict_proba(X)
        prediction[prediction == 0] += np.finfo(prediction.dtype).tiny
        return np.log(prediction)
