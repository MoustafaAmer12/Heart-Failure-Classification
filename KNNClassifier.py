from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.knn.fit(X_scaled, y)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.knn.predict(X_scaled)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"{self.k}-NN Model Accuracy: {accuracy:.2f}")
        return accuracy
    

if __name__ == "__main__":
    random_seed = 42
    X, y = make_classification(n_samples=500, n_features=10, random_state=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNNClassifier(k=10)
    knn.fit(X_train, y_train)
    knn.evaluate(X_test, y_test)
