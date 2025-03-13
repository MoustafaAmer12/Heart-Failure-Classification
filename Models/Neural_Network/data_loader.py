import torch
import torch.utils.data as data

class HeartFailureDataset(data.Dataset):
    def __init__(self, X, y, device=None):
        self.device = device if device else torch.device("cpu")
        self.X = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        self.y = torch.tensor(y.values, dtype=torch.float32).squeeze().to(self.device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loader(X_df, y_df, batch_size=32, shuffle=True, device=None):
    dataset = HeartFailureDataset(X_df, y_df, device=device)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
