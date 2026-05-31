import numpy as np
#logistic regression does not give us number/value it gives us catagories.
class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.lr = learning_rate
        self.epoch = epochs
        self.weight = None
        self.bias = None
        self.loss_history = []
    # add sigmoid function - the curve line.
    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.weight = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epoch):
            z = np.dot(X, self.weight) + self.bias
            prediction = self.sigmoid(z)
            dw = (1/n_samples) * np.dot(X.T, prediction - y)
            db = (1/n_samples) * np.sum(prediction - y)
            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
            loss = -np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
            self.loss_history.append(loss)

    def predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        probablities = self.sigmoid(z)
        return np.where(probablities >= 0.5, 1, 0)
    
    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Loss over time - Logistic Regression')
        plt.show()