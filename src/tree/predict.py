def predict_sample(node, x):
    while not node.is_leaf:
        val = x.get(node.attribute)

        if val is None:
            return node.majority_class

        if node.threshold is not None:
            node = node.left if val <= node.threshold else node.right
        else:
            node = node.left if val in node.split else node.right

    return node.prediction


def predict(tree, X):
    return [predict_sample(tree.root, row) for _, row in X.iterrows()]
