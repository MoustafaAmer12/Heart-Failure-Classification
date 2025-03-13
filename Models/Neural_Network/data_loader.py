import torch.utils.data as data  

class HeartFailureDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)  # Number of samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_data_loader(X_tensor, y_tensor):
    dataset = HeartFailureDataset(X_tensor, y_tensor)
    # Use DataLoader to load data in batches
    return data.DataLoader(dataset, batch_size=32, shuffle=False)