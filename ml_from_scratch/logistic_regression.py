import numpy as np
#logistic regression does not give us number/value it gives us catagories.
class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.lr = learning_rate
        self.epoch = epochs
        self.weight = None
        self.bias = None
    # add sigmoid function - the curve line.
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.weight = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epoch):
            z = np.dot(X, self.weight) + self.bias
            pridiction = self.sigmoid(z)
            dw = (1/n_samples) * np.dot(X.T, pridiction - y)
            db = (1/n_samples) * np.sum(pridiction - y)
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        probablities = self.sigmoid(z)
        return np.where(probablities >= 0.5, 1, 0)