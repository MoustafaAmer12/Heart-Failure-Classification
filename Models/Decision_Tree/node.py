import numpy as np

class Node:
    """
    Parameters
    ----------
    data: numpy.ndarray, default=None
        The dataset includes X and Y
    children: dict(feat_value: Node), default=None
        Dict of children
    split_on: int, default=None
        Index of the feature that node was split on that
    pred_class : str, default=None
        The predicted class for the node (only applicable to leaf nodes)
    is_leaf: bool, default=False
        Determine whether the node is leaf or not
    split_val: float, default=None
        Determines the value to split on
    """

    def __init__(self, data=None, children:list=None, split_on=None, is_leaf=False, pred_class=None, split_val=None):
        self.data = data
        self.children = children
        self.pred_class = pred_class
        self.is_leaf = is_leaf
        self.split_val = split_val