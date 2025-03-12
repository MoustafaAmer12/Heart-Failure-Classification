
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, n_estimators = 150):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        np.random.seed(42)
        n_samples, n_features = X.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y)

            predictions = model.predict(X)

            err = np.sum(weights * (predictions != y)) / np.sum(weights)

            if err > 0.5:
                continue

            alpha = 0.5 * np.log((1 - err) / err)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights) # Normalize weights

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)

        return np.sign(final_pred)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate synthetic dataset
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    y = 2 * (y - 0.5)  # Convert to {-1, 1} labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train AdaBoost model
    model = AdaBoost(n_estimators=50)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    print("Accuracy:", accuracy_score(y_test, y_pred))
