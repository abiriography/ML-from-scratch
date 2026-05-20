import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kmeans import KMeans

# Create data
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# Your model
model = KMeans(k=3, epochs=100)
model.fit(X)
predictions = model.predict(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], 
            c='red', marker='x', s=200, linewidths=3)
plt.title('KMeans Clustering from Scratch')
plt.show()