# 🤖 ML From Scratch

> Implementing core Machine Learning algorithms from scratch using only Python and NumPy — no Scikit-learn, no shortcuts.

---

## 📌 About

This repository is a personal deep-dive into how Machine Learning algorithms actually work under the hood. Each algorithm is built from first principles, then benchmarked against Scikit-learn to validate correctness.

---

## 📂 Algorithms Implemented

### 1. 📈 Linear Regression

**What it does:** Predicts a continuous value from input features using a learned weight and bias.

**How it works:**
- Fits a line `y = wx + b` to the training data
- Uses **Gradient Descent** to minimize the loss by iteratively moving in the direction that reduces error
- Loss is measured using **Mean Squared Error (MSE)**

**Bug found & fixed:** Forgot to wrap the weight update inside `for epoch in range(self.epochs)` — without the loop, the model only trained for one step, causing inconsistent MSE results.

| Metric | Score |
|--------|-------|
| My MSE | 312.28 |
| Sklearn MSE | 316.99 |

---

### 2. 🔵 Logistic Regression

**What it does:** Predicts a **category** (e.g. yes/no, spam/not spam) instead of a numeric value.

**How it works:**
- Passes a linear combination of inputs through the **Sigmoid function**
- Sigmoid maps any real number to a value between 0 and 1, which is interpreted as a probability
- A threshold (usually 0.5) converts the probability into a class label

| Metric | Score |
|--------|-------|
| My Accuracy | 87% |
| Sklearn Accuracy | 90% |

---

### 3. 🟡 K-Means Clustering

**What it does:** Groups unlabeled data points into `k` clusters based on similarity.

**How it works:**
- Picks 3 random points as initial cluster centroids
- Assigns every data point to the nearest centroid (using Euclidean distance)
- Recalculates the centroid as the mean of all assigned points
- Repeats until the centroids stop moving

**Plot output:** 3 color-coded groups of data points with red ✕ marks showing the center of each cluster.

---

### 4. 📧 Naive Bayes

**What it does:** A probabilistic classifier commonly used for tasks like **spam detection**.

**How it works:**
- Analyzes words in an email (or text)
- Calculates the probability of each class (spam / not spam) given the observed words
- Uses Bayes' Theorem, assuming all features (words) are independent of each other
- Predicts the class with the highest probability

---


---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| NumPy | All mathematical operations |
| Matplotlib | Visualizations (K-Means plot) |
| Scikit-learn | Benchmarking only |

---

## 🚀 Getting Started


# Clone the repo
git clone https://github.com/abiriography/ML-from-scratch.git
cd ML-from-scratch

# Install dependencies
pip install numpy matplotlib scikit-learn

# Run any algorithm
python linear_regression.py
python logistic_regression.py
python kmeans.py
python naive_bayes.py
python neural_network.py
```

## 🎯 Goal

The goal isn't just to get the right answer — it's to understand **why** the answer is right. By building each algorithm by hand and comparing against Scikit-learn, every implementation is both a learning exercise and a validation.

---

## 👤 Author

**abiriography** — building ML one algorithm at a time.
