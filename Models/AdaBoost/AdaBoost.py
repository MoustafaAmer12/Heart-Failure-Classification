
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from Data_Preprocessing.preprocess_data import PrepareData



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
            model.fit(X, y, sample_weight=weights)  # Pass weights to fit

            predictions = model.predict(X)

            err = np.sum(weights * (predictions != y)) / np.sum(weights)

            if err >= 0.5:
                break 

            alpha = 0.5 * np.log((1 - err) / err)

            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize weights

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        final_pred = np.zeros(X.shape[0])

        for alpha, model in zip(self.alphas, self.models):
            final_pred += alpha * model.predict(X)

        return np.where(final_pred >= 0, 1, -1)  # Ensure valid class labels
    

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        print(f"AdaBoost Model Accuracy: {accuracy:.2f}")
        print(f"AdaBoost Model F1-Score: {f1:.2f}")
        return accuracy, f1

    
    def plot_confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for AdaBoost")
        plt.show()



if __name__ == "__main__":
    random_seed = 42

    DataPrep = PrepareData(dataset_path='heart.csv', random_seed=random_seed
                           ,training_percentage=70, validation_percentage=10, testing_percentage=20)
    
    X_train, X_validation, X_test, y_train, y_validation, y_test = DataPrep.prepare_data()

    y_train = 2 * (y_train - 0.5)
    y_test = 2 * (y_test - 0.5)
    y_validation = 2 * (y_validation - 0.5)

    # Train AdaBoost model
    model = AdaBoost(n_estimators=50)
    model.fit(X_train, y_train)

    test_accuracy, test_f1 = model.evaluate(X_test, y_test)
    validation_accuracy, validation_f1 = model.evaluate(X_validation, y_validation)

    # model.plot_confusion_matrix(X_test, y_test)
    # model.plot_confusion_matrix(X_validation, y_validation)
    
    # Summary Output
    print(f"Test Summary:\nAccuracy: {test_accuracy:.2f}\nF1-Score: {test_f1:.2f}")
    print(f"Validation Summary:\nAccuracy: {validation_accuracy:.2f}\nF1-Score: {validation_f1:.2f}")
