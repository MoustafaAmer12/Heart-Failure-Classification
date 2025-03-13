import torch
import torch.nn as nn  
import torch.optim as optim  
from nn import HeartFailureNN
from data_loader import get_data_loader
from Data_Preprocessing.preprocess_data import PrepareData

DATA_PATH = "Data/heart.csv"
class NNClassifier:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training function
    def train(self, train_loader, epochs=50):
        self.model.train()  # Set model to training mode
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()  # Reset gradients
                outputs = self.model(batch_X)  # Forward pass
                loss = self.criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, test_loader):
        self.model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No need to track gradients during evaluation
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                predicted = (outputs >= 0.5).float()  # Convert probabilities to 0/1
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    neural_net = HeartFailureNN()
    model = NNClassifier(neural_net)
    
    prepared_data = PrepareData(dataset_path=DATA_PATH,random_seed=42,
                                training_percentage=70,validation_percentage=10,testing_percentage=20)
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepared_data.prepare_data()
    train_loader = get_data_loader(X_train, y_train)
    test_loader = get_data_loader(X_test, y_test)
    
    # test_loader = get_data_loader()

    print(neural_net)
    # model.train(train_loader, epochs=50)
    # model.evaluate(test_loader)