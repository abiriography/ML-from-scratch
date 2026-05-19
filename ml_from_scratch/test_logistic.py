import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as SklearnLR
from logistic_regression import LogisticRegression

# Create data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Your model
my_model = LogisticRegression(learning_rate=0.1, epochs=1000)
my_model.fit(X_train, y_train)
my_predictions = my_model.predict(X_test)
print("Your accuracy:", accuracy_score(y_test, my_predictions))

# Sklearn
sk_model = SklearnLR()
sk_model.fit(X_train, y_train)
sk_predictions = sk_model.predict(X_test)
print("Sklearn accuracy:", accuracy_score(y_test, sk_predictions))