import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from Naive_bayes import NaiveBayes

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Your model
my_model = NaiveBayes()
my_model.fit(X_train, y_train)
my_predictions = my_model.predict(X_test)
print("Your accuracy:", accuracy_score(y_test, my_predictions))

# Sklearn
sk_model = GaussianNB()
sk_model.fit(X_train, y_train)
sk_predictions = sk_model.predict(X_test)
print("Sklearn accuracy:", accuracy_score(y_test, sk_predictions))