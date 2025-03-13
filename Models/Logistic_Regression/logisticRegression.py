import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from Data_Preprocessing.preprocess_data import PrepareData

class LogisticRegression:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = SklearnLogisticRegression(max_iter=1000, random_state=self.random_state)
        self.best_model = None
        self.feature_names = None
    
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self
    
    def evaluate(self, X, y, dataset_name=""):
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"Confustion matrix: {cm}")


# Example usage
if __name__ == "__main__":

    data_prep = PrepareData(
        dataset_path='heart.csv',
        random_seed=42,
        training_percentage=70,
        validation_percentage=10,
        testing_percentage=20
    )
    
    # Prepare the data
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.prepare_data()
    
    if X_train is not None:
        # Initialize the model
        model = LogisticRegression(random_state=42)
                
        # Train the initial model
        print("\n--- Training Initial Model ---")
        model.train(X_train, y_train)
        
        # Evaluate the initial model
        print("\n--- Initial Model Evaluation ---")
        model.evaluate(X_val, y_val, "Validation")
    
    else:
        print("Data preparation failed. Please check the dataset path and try again.")