import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression

# Create simple data
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Test your model
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("First 5 predictions:", predictions[:5])
print("First 5 actual values:", y_test[:5])
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error

sk_model = SklearnLR()
sk_model.fit(X_train, y_train)
sk_predictions = sk_model.predict(X_test)

print("Your MSE:", mean_squared_error(y_test, predictions))
print("Sklearn MSE:", mean_squared_error(y_test, sk_predictions))