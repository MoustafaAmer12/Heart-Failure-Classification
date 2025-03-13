from node import Node
import numpy as np

class DecisionTreeClassifier:
    """
    Decision Tree Classifier Class
    Constructs a generic decision tree for classification
    """
    def __init__(self, X, Y, max_depth=None, min_samples_split=None):
        self.root = Node(data=np.column_stack((X, Y)))
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    
        
    def train(self):
        """
        Trains the decision tree model to the provided dataset.
        Done by constructing the decision tree.

        Parameters:
        -----------
        data: numpy.ndarray
            The dataset.

        max_depth: int
            The maximum depth of the tree.
        
        min_samples_split: int
            The minimum number of samples existing in a node to stop.
        """
        self.best_split(self.root)

    def predict(self, test):
        """
        Predict the class labels for the given input features.

        Parameters:
        -----------
        test: numpy.ndarray
            The input features for which to make predictions. Should be a 2D array-like object.

        Returns:
        -----------
        predictions: numpy.ndarray
            An array of predicted class labels.

        """
        # Traverse the decision tree for each input and make predictions
        predictions = np.array([self.traverse_tree(sample, self.root) for sample in test])
        return predictions

    def traverse_tree(self, sample, node):
        """
        Recursively traverse the decision tree to predict the class label for a given input.

        Parameters:
        -----------
        sample:
            The input for which to make a prediction.

        node:
            The current node being traversed in the decision tree.

        Returns:
        -----------
        predicted_class:
            The predicted class label for the input feature.

        """

        # Check if the current node is a leaf node
        if node.is_leaf:
            return node.pred_class

        # Get the feature value at the split point for the current node
        feat_value = sample[node.split_on]

        # Recursively traverse the decision tree using the child node matching feature value
        # 0 if feat_val < split_val, else 1
        predicted_class = (
            self.traverse_tree(sample, node.children[0]) 
            if feat_value <= node.split_val 
            else self.traverse_tree(sample, node.children[1])
        )

        return predicted_class

    def split_on_feature(self, data, feat_index):
        """
        Split the dataset based on a specific feature index.

        - Determines Splt points by iterating over all data points
        and performing a split on each midpoint then returns the best
        splitting threshold

        Parameters:
        -----------
        data: numpy.ndarray
            The dataset to be split.

        feat_index: int
            The index of the feature to perform the split.

        Returns:    
        -----------
        - split_nodes: dict
            A dictionary of split nodes. 
            (feature value as key, corresponding node as value)

        - threshold: float
            Threshold for performing the slit
 
        - weighted_entropy: float
            The weighted entropy of the split.
        """
        # Get sorted unique values to determine the split point
        sorted_unique_values = np.sort(np.unique(data[:, feat_index]))

        split_nodes = {}
        min_entropy = 999999
        split_threshold = -1  # For Normalized Data no split

        total_instances = len(data)
        
        for i in range(1, len(sorted_unique_values)):
            weighted_entropy = 0
            # Compute midpoint between consecutive unique values
            threshold = (sorted_unique_values[i - 1] + sorted_unique_values[i]) / 2

            # Create First Data Partition (node)
            partition_1 = data[data[:, feat_index] <= threshold, :]
            node_1 = Node(partition_1)
            partition_y1 = self.get_y(partition_1)
            node_1_entropy = self.calculate_entropy(partition_y1)
            weighted_entropy += (len(partition_1) / total_instances) * node_1_entropy

            # Create Second Data Partitio (node)
            partition_2 = data[data[:, feat_index] > threshold, :]
            node_2 = Node(partition_2)
            partition_y2 = self.get_y(partition_2)
            node_2_entropy = self.calculate_entropy(partition_y2)
            weighted_entropy += (len(partition_2) / total_instances) * node_2_entropy

            if weighted_entropy < min_entropy:
                split_nodes[0] = node_1
                split_nodes[1] = node_2
                min_entropy = weighted_entropy
                split_threshold = threshold

        return split_nodes, split_threshold, min_entropy
    
    @staticmethod
    def cont_split_point(feature_values):
        np.argsort(feature_values)


    def best_split(self, node, depth=0):
        """
        Find the best split for the given node.

        Parameters:
        ----------
        node: Node
            The node that carries data for which the best split is being determined.

        If the node meets the criteria to stop splitting:
            - Mark the node as a leaf.
            - Assign a predicted class for predictions based on the target values (y).
            - return.

        Otherwise:
            - Iterate over the features to find the best split.
            - Split the data based on each feature and calculate the weighted entropy of the split.
            - Compare the current weighted entropy with the previous best entropy.
            - Update the best split variables if the current split has lower entropy.
            - update the node with the best split information, including child nodes and the feature index used for the split.
            - Recursively call the best_split function for each child node.

        """
        # Base Case if the node meets the criteria to stop splitting
        if self.meet_criteria(node, depth):
            node.is_leaf = True
            y = self.get_y(node.data)
            node.pred_class = self.get_pred_class(y)
            return

        # Initialize variables for tracking the best split
        index_feature_split = -1
        min_entropy = 1

        # iterate over all features, ignore (y)
        for i in range(node.data.shape[1] - 1):
            split_nodes, threshold, weighted_entropy = self.split_on_feature(node.data, i)
            if weighted_entropy < min_entropy:
                child_nodes, split_val , min_entropy = split_nodes, threshold, weighted_entropy
                index_feature_split = i

        node.children = child_nodes
        node.split_on = index_feature_split
        node.split_val = split_val

        # Recursively call the best_split function for each child node
        for child_node in child_nodes.values():
            self.best_split(child_node, depth + 1)

    def meet_criteria(self, node:Node, depth):
        """
        Check if the criteria for stopping the tree expansion is met for a given node. 
        Here we only check if the entropy of the target values (y) is zero.
        Parameters:
        -----------
        node : Node
            The node to check for meeting the stopping criteria.

        Returns:
        -----------
        bool
            True if the criteria is met, False otherwise.

        """
        if self.max_depth and depth >= self.max_depth:
            return True

        y = self.get_y(node.data)
        if self.min_samples_split and len(y) <= self.min_samples_split:
            return True
        return True if self.calculate_entropy(y) == 0 else False
    

    @staticmethod
    def get_y(data:np.ndarray):
        """
        Get the target (y) from the data.

        Parameters:
        -----------
        data : numpy.ndarray
            The input data containing features and the target variable.

        Returns:
        -----------
        y: numpy.ndarray
            The target variable extracted from the data.

        """
        return data[:, -1]
    

    @staticmethod
    def get_pred_class(y):
        """
        Get the predicted class label based on the majority vote.

        Parameters:
        -----------
        Y : numpy.ndarray
            The array of class labels.

        Returns:
        -----------
        str
            The predicted class label.

        """

        labels, labels_counts = np.unique(y, return_counts=True)
        return labels[np.argmax(labels_counts)]

    @staticmethod
    def calculate_entropy(y:np.array):
        """
        Calculates Entropy based on the summation rule
        of pi * log2(1/pi)

        Args:
            y (np.array): array of class value (0 or 1)
        Returns:
            double: entropy value
        """
        total_instances = len(y)
        _, label_counts = np.unique(y, return_counts=True)
        entropy = sum([label_count/total_instances * np.log2(total_instances / label_count) for label_count in label_counts])
        return entropy



if __name__ == "__main__":
    data = np.array([
        [1 , 1, 1, 1],
        [0 , 0, 1, 1],
        [0 , 1, 0, 0],
        [1 , 0, 1, 0],
        [1 , 1, 1, 1],
        [1 , 1, 0, 1],
        [0 , 0, 0, 0],
        [1 , 1, 0, 1],
        [0 , 1, 0, 0],
        [0 , 1, 0, 0],
    ])
    X = data[:, :-1]
    Y = data[:, -1]
    model = DecisionTreeClassifier(X, Y)
    model.train()
    pred = model.predict([
        [1 , 1, 1],
        [0 ,0, 1],
        [0 , 1, 0]
    ])
    print(pred)
