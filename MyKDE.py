import math
import numpy as np

class MyKDE:
    def __init__(self, width):
        self.width = width
    
    def gaussian(self, x):
        return 1/np.sqrt(2*math.pi)*np.exp(-0.5*(x**2))

    def fit(self, X):
        X = np.array(X)
        X0 = np.arange(np.floor(np.min(X)), np.ceil(np.max(X)), 0.001)
        yhat = np.zeros(len(X0))
        for i in range(len(X0)):
            yhat[i] = np.sum(self.gaussian(abs(X-X0[i])/self.width))/(len(X)*self.width)
        fig, ax = plt.subplots()
        ax.plot(X0, yhat)
        ax.set_ylabel('Density')
        plt.show()
