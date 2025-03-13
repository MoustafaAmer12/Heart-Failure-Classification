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

    Examples
    --------
    >>> feat_index = 0     # Ear Shape
    >>> root = Node(data=all_data, split_on=feat_index)
    >>> pointy_node = Node(data=pointy_data, is_leaf=True)
    >>> floppy_node = Node(data=floppy_data, is_leaf=True)
    >>> root.children = {"Pointy": pointy_node, "Floppy": floppy_node}

    Visualization
    -------------
                                 root  (data = all_data, split_on = 0, is_leaf=False)
                                /    \
                               /      \
                              /        \
                             /          \
                     pointy_node     floppy_node
    (data=pointy_data, is_leaf=True)    (data=floppy_data, is_leaf=True)
    """

    def __init__(self, data=None, children:list=None, split_on=None, is_leaf=False, pred_class=None):
        self.data = data
        self.children = children
        self.pred_class = pred_class
        self.is_leaf = is_leaf