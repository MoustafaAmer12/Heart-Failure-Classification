from Data_Preprocessing.preprocess_data import PrepareData

DATA_PATH = "Data/heart.csv"
if __name__ == "__main__":
    prepared_data = PrepareData(dataset_path=DATA_PATH,random_seed=42,
                                training_percentage=70,validation_percentage=10,testing_percentage=20)
    X_train, X_val, X_test, y_train, y_val, y_test = prepared_data.prepare_data()
    print(X_train.head())
    print(X_val.head())
    print(X_test.head())
    print(y_train.head())
    print(y_val.head())
    print(y_test.head())