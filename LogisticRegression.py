import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.001, n=1000):
        # Constructor to initialize learning rate and number of iterations
        self.lr = lr
        self.n = n
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        # Fit the logistic regression model to the training data
        X = np.insert(X, 0, 1, axis=1)
        weights = np.zeros(X.shape[1])

        for i in range(self.n):
            # Compute predictions using the sigmoid function
            y_pred = self._sigmoid(np.dot(X, weights))
            # Compute gradient for weights
            gradient_min = np.dot((y - y_pred), X) / X.shape[0]
            # Update weights
            weights += self.lr * gradient_min

        # Store intercept and weights
        self.intercept = weights[0]
        self.weights = weights[1:]

    def _predict(self, X):
        # Predict labels for new data
        X = np.insert(X, 0, 1, axis=1)
        # Compute predictions
        y_sigmoid_pred = self._sigmoid(np.dot(X, np.insert(self.weights, 0, self.intercept)))
        # Return class labels based on sigmoid output
        return [1 if y_sigmoid_pred > 0.5 else 0 for y_sigmoid_pred in y_sigmoid_pred]

    def _sigmoid(self, X):
        # Compute sigmoid function
        return 1 / (1 + np.exp(-X))
