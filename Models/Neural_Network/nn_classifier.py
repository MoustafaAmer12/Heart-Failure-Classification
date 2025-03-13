import torch
import torch.nn as nn  
import torch.optim as optim  
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nn import HeartFailureNN
from data_loader import get_data_loader
from preprocess_data import PrepareData

DATA_PATH = "Data/heart.csv"
LEARNING_RATE = 0.008
EPOCHS = 50
BATCH_SIZE = 32
RANDOM_SEED = 42
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2

class NNClassifier:
    def __init__(self, model, learning_rate=LEARNING_RATE):
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    def train(self, train_loader, epochs=EPOCHS):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    def evaluate(self, data_loader, set_name="Test"):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                outputs = self.model(batch_X)
                predicted = (outputs >= 0.5).float()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"{set_name} Accuracy: {accuracy:.2f}%")
        print(f"{set_name} F1 Score: {f1:.4f}")
        print(f"{set_name} Confusion Matrix:\n{cm}")
        
        # Plot Confusion Matrix
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{set_name} Confusion Matrix")
        plt.show()

if __name__ == "__main__":
    prepared_data = PrepareData(dataset_path=DATA_PATH, random_seed=RANDOM_SEED,
                                training_percentage=TRAIN_SPLIT*100,
                                validation_percentage=VAL_SPLIT*100,
                                testing_percentage=TEST_SPLIT*100,)
    X_train, X_val, X_test, y_train, y_val, y_test = prepared_data.prepare_data()

    train_loader = get_data_loader(X_train, y_train, batch_size=BATCH_SIZE)
    val_loader = get_data_loader(X_val, y_val, batch_size=BATCH_SIZE)
    test_loader = get_data_loader(X_test, y_test, batch_size=BATCH_SIZE)

    model = NNClassifier(HeartFailureNN())
    model.train(train_loader, epochs=EPOCHS)
    
    # Evaluate on validation and test sets
    model.evaluate(val_loader, set_name="Validation")
    model.evaluate(test_loader, set_name="Test")
