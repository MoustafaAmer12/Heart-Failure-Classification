import numpy as np
from sklearn.tree import DecisionTreeClassifier
from Data_Preprocessing.preprocess_data import PrepareData
from Models.Decision_Tree.decision_tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


class Bagging:
    def __init__(self, X, Y, random_seed):
        self.X = X
        self.Y = Y
        self.random_seed = random_seed
        self.models = []
    
    def resampling(self, m=5):
        X_bootstrap_samples = []
        Y_bootstrap_samples = []

        np.random.seed(self.random_seed)
        n = len(self.X)  # Number of rows in training data

        for i in range(m):
            np.random.seed(self.random_seed + i)
            indices = np.random.choice(n, n, replace=True)

            X_sample, Y_sample = self.X.iloc[indices], self.Y.iloc[indices]

            X_bootstrap_samples.append(X_sample)
            Y_bootstrap_samples.append(Y_sample)

        return X_bootstrap_samples, Y_bootstrap_samples

    
    def train_models(self, X_samples, Y_samples):
        for i in range(len(X_samples)):
            model = DecisionTreeClassifier(X_samples[i], Y_samples[i])
            model.train()
            self.models.append(model)

    def ensemble_models(self, X_test):
        predictions = np.array([model.predict(X_test) for model in self.models])

        #Majority vote (Average and threshold)
        avg_predictions = np.mean(predictions, axis = 0)
        bagged_predictions = (avg_predictions > 0.5).astype(int)
        
        return bagged_predictions
    
    def predict(self, X_test):
        #Resampling 
        X_samples, Y_samples = self.resampling()

        #Train models
        self.train_models(X_samples, Y_samples)

        #Ensembling models
        bagged_prediction = self.ensemble_models(X_test)

        return bagged_prediction

    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy_val = accuracy_score(y_test, y_pred)
        
        # Calculate F1 score
        f1_score_val = f1_score(y_test, y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy_val,
            'f1_score': f1_score_val,
            'confusion_matrix': cm
        }
        
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

    bagging_clf = Bagging(X_train, y_train, random_seed=42)

    evaluation_results = bagging_clf.evaluate(X_test, y_test)

    print("===============================")
    print("Evaluation Results:")
    print("===============================")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
    print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(evaluation_results['confusion_matrix'])
    
    # Plot confusion matrix






            



