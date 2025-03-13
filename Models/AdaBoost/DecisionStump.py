import numpy as np
import pandas as pd

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1  # 1 means x >= threshold -> 1, otherwise -1
        self.alpha = None  # Weight of this stump in AdaBoost

    def train(self, X, y, sample_weights):
        """
        Trains the decision stump by finding the best feature and threshold 
        to split on, minimizing weighted classification error.
        """
        import pandas as pd
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X  # Convert to NumPy
        y = y.to_numpy() if isinstance(y, pd.Series) else y

        n_samples, n_features = X.shape
        min_error = float('inf')

        for feature_index in range(n_features):
            feature_values = np.sort(np.unique(X[:, feature_index]))

            for i in range(1, len(feature_values)):
                threshold = (feature_values[i - 1] + feature_values[i]) / 2

                for polarity in [1, -1]:
                    predictions = np.ones(n_samples) * polarity
                    predictions[X[:, feature_index] < threshold] = -polarity

                    weighted_error = np.sum(sample_weights[y != predictions])

                    if weighted_error < min_error:
                        min_error = weighted_error
                        self.feature_index = feature_index
                        self.threshold = threshold
                        self.polarity = polarity

    def predict(self, X):
        import pandas as pd
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X  # Convert to NumPy

        n_samples = X.shape[0]
        predictions = np.ones(n_samples) * self.polarity
        predictions[X[:, self.feature_index] < self.threshold] = -self.polarity  # Now works

        return predictions
