from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class PrepareData:
    def __init__(self, dataset_path, random_seed
                  ,training_percentage, validation_percentage, testing_percentage):
        self.dataset_path = dataset_path
        self.random_seed = random_seed
        self.training_percentage = training_percentage
        self.validation_percentage = validation_percentage
        self.testing_percentage = testing_percentage
        self.X_train = None
        self.X_validation = None
        self.X_test = None
        self.Y_train = None
        self.Y_validation = None
        self.Y_testing = None

    def load_data(self):
        try:
            df = pd.read_csv(self.dataset_path)
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def split_data(self, df, target_column = 'HeartDisease'):
        #Setting the random seed
        np.random.seed(self.random_seed)

        #Setting splitting sizes
        test_size = self.testing_percentage/100.0
        validation_size = self.validation_percentage/(100.0-self.testing_percentage)

        print(f"Splitting sizes:\ntest size:{test_size}\nvalidation size:{validation_size}")    
        #Splitting target column from features
        X = df.drop(columns = [target_column])
        Y = df[target_column]

        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        print("Categorical Features:", categorical_features )

        # One-hot encode the categorical features
        X = pd.get_dummies(X, columns=categorical_features)

        X = X.astype(int)

        # Split data into 80% for training and validation, and 20% for testing
        X_temp, self.X_test, Y_temp, self.Y_testing = train_test_split(
            X,Y, test_size=test_size, random_state=self.random_seed, stratify=Y
        )

        #Split the data of training and validation to 70% training and 10% validation
        #From the 80% interval
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(
            X_temp, Y_temp, test_size=validation_size, random_state=self.random_seed, stratify=Y_temp
        )

        total_samples = len(df)
        train_proportion = len(self.X_train) / total_samples
        val_proportion = len(self.X_validation) / total_samples
        test_proportion = len(self.X_test) / total_samples
        
        print(f"Data split completed:")
        print(f"Training set: {len(self.X_train)} samples ({train_proportion:.1%})")
        print(f"Validation set: {len(self.X_validation)} samples ({val_proportion:.1%})")
        print(f"Test set: {len(self.X_test)} samples ({test_proportion:.1%})")

        return self.X_train, self.X_validation, self.X_test, self.Y_train, self.Y_validation, self.Y_testing

    def prepare_data(self):
        df = self.load_data()
        if df is not None:
            return self.split_data(df)
        else:
            return None
