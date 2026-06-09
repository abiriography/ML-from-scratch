import numpy as np
class NeuralNetwork:
    def __init__(self,learning_rate = 0.01, epochs = 1000, hidden_size = 64):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_size = hidden_size
    
    def sigmoid(self, z):
        return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))
    
    def sigmoid_derivatives(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
                           
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize weigths randomly, not zeros
        self.W1 = np.random.randn(n_features, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, 1) * 0.01
        self.b2 = np.zeros((1,1))

        for epoch in range(self.epochs):
            # forward propagation 
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)

            #backpropagation
            dA2 = A2 - y.reshape(-1,1)
            dW2 = np.dot(A1.T, dA2) / n_samples
            db2 = np.sum(dA2, axis=0) / n_samples
            dA1 = np.dot(dA2, self.W2.T) * self.sigmoid_derivatives(Z1)
            dW1 = np.dot(X.T, dA1) / n_samples
            db1 = np.sum(dA1, axis=0) / n_samples

            #update weights
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
    
    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        return np.where(A2>= 0.5, 1, 0)