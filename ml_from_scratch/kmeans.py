import numpy as np
class KMeans:
    def __init__(self, k = 3, epochs = 100):
        self.k = k
        self.epochs = epochs
        self.centroids = None
    def fit(self, X):
        random_indices = np.random.choice(X.shape[0], self.k, replace = False)
        self.centroids = X[random_indices]

        for epoch in range(self.epochs):
            labels = self._assign_clusters(X)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
            if np.allclose(self.centroids, new_centroids):
                 break
            self.centroids = new_centroids

    def _assign_clusters(self, X):
            labels = []
            for point in X:
                distance = [np.sqrt(np.sum((point - centroid)**2)) for centroid in self.centroids]
                nearest = np.argmin(distance)
                labels.append(nearest)
            return np.array(labels)
    def predict(self, X):
        return self._assign_clusters(X)