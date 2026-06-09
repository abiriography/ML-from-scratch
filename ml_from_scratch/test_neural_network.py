import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Neural_Network import NeuralNetwork

# Create data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your model
model = NeuralNetwork(learning_rate=0.1, epochs=1000, hidden_size=64)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Your accuracy:", accuracy_score(y_test, predictions.flatten()))