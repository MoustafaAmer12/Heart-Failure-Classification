import torch
import torch.nn as nn  
import torch.optim as optim  
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

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                predicted = (outputs >= 0.5).float()  # Convert probabilities to binary
                
                # Ensure the shape matches
                batch_y = batch_y.view_as(predicted)  # Reshape if needed
                
                correct += (predicted == batch_y).sum().item()
                total += batch_y.numel()  # Use numel() to ensure correct counting
        
        accuracy = (correct / total) * 100
        print(f"Test Accuracy: {accuracy:.2f}%")

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
    model.evaluate(test_loader)
