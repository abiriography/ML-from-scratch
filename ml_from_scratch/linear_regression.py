import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_sample = X.shape[0]
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            pridiction = np.dot(X, self.weights) + self.bias #equation of line
            loss = np.mean((pridiction - y)**2)
            dw = (2/n_sample) * np.dot(X.T, (pridiction - y))
            db = (2/n_sample) * np.sum(pridiction - y)
            self.weights = self.weights - self.lr * dw
        self.bias = self.bias - self.lr * db

    def predict(self, X):
        pridiction = np.dot(X, self.weights) + self.bias        
        return np.dot(X, self.weights) + self.bias