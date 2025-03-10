import numpy as np
from sklearn.tree import DecisionTreeClassifier

class Bagging:
    def __init__(self, X, Y, random_seed):
        self.X = X
        self.Y = Y
        self.random_seed = random_seed
        self.models = []
    
    def resampling(self, m=50):
        #m => number of independent bootstrap models
        X_bootstrap_samples = []
        Y_bootstrap_samples = []
        
        np.random.seed(self.random_seed)

        #Number of examples = bootstrap_sample_size => n
        n = len(self.X)
        
        #Resampling
        for i in range(m):
            np.random.seed(self.random_seed+i)
            #Randomly sample with replacement n indices in training data set
            indices = np.random.choice(n, n, replace=True)

            #Get sample with size n from the training data
            X_sample, Y_sample = self.X[indices],self.Y[indices]
                
            X_bootstrap_samples.append(X_sample)
            Y_bootstrap_samples.append(Y_sample)

        return X_bootstrap_samples,Y_bootstrap_samples
    
    def train_models(self, X_samples, Y_samples):
        for i in range(len(X_samples)):
            model = DecisionTreeClassifier()
            model.fit(X_samples[i],Y_samples[i])
            self.models.append(model)

    def ensemble_models(self, X_test):
        predictions = np.array([model.predict(X_test) for model in self.models])

        #Majority vote (Average and threshold)
        avg_predictions = np.mean(predictions, axis = 0)
        bagged_predictions = (avg_predictions > 0.5).astype(int)
        return bagged_predictions

        
if __name__ == "__main__":
    #Hardcoded random data set of 100 examples and binary classes
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100) #Binary class {0,1} for the 100 examples
    
    X_test = np.random.rand(10, 5)
    bagging_clf = Bagging(X_train, y_train, random_seed=42)

    #Resampling 
    X_samples, Y_samples = bagging_clf.resampling()

    #Train models
    bagging_clf.train_models(X_samples, Y_samples)

    #Ensembling models
    bagged_prediction = bagging_clf.ensemble_models(X_test)

    print(bagged_prediction)





            



