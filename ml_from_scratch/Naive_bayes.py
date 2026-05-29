import numpy as np
class NaiveBayes:
    def __init__(self):
        pass
    def fit(self, X,y):
        n_samples, n_feature = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        #initialize mean, varience, prior
        self.mean = np.zeros((n_classes, n_feature))
        self.varience = np.zeros((n_classes, n_feature))
        self.prior = np.zeros(n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i] = X_c.mean(axis=0)
            self.varience[i] = X_c.var(axis=0)
            self.prior[i] = X_c.shape[0] / n_samples

    def _gaussian_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        variance = self.varience[class_idx]
        numerator = np.exp(-((x - mean)**2) / (2* variance))
        denominator = np.sqrt(2* np.pi * variance)
        return numerator / denominator
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for i, c in enumerate(self.classes):
                prior = np.log(self.prior[i])
                likelihood = np.sum(np.log(self._gaussian_pdf(i, x)))
                posteriors.append(prior + likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)