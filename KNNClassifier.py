
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter



class KNNClassifier:
    def __init__(self, k = 3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = y

    def predict(self, X):
        X = self.scaler.transform(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def _predict_single(self, x):

        distances = np.linalg.norm(self.X_train - x, axis = 1)

        k_neighbours = np.argsort(distances)[:self.k]

        k_labels = self.y_train[k_neighbours]

        most_common = Counter(k_labels).most_common(1)

        return most_common[0][0]
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"{self.k}-NN Model Accuracy: {accuracy:.2f}")
        return accuracy
    



if __name__ == "__main__":
    random_seed = 42
    X, y = make_classification(n_samples=500, n_features= 10, random_state= random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNNClassifier(k = 10)

    knn.fit(X_train, y_train)

    knn.evaluate(X_test, y_test)

