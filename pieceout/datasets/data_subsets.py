import numpy as np

from pylearn2.utils.one_hot import compressed_one_hot
from pylearn2.datasets import DenseDesignMatrix


class ClassificationSubtask(DenseDesignMatrix):
    def __init__(self, dataset, classes, one_hot=True,
                 simplify_binary=True):
        X = dataset.get_design_matrix()
        y = dataset.get_targets()
        assert set(classes).issubset(set(y))
        mask = (y.reshape((-1, 1)) == np.asarray(classes)).any(axis=1)
        X_ = X[mask]
        y_ = y[mask]
        if one_hot:
            y_, _ = compressed_one_hot(y_)
        else:
            for i, c in enumerate(classes):
                y_[y_ == c] = i
        super(ClassificationSubtask, self).__init__(X=X_, y=y_)


class Bagged(DenseDesignMatrix):
    def __init__(self, dataset, classes, one_hot=True):
        X = dataset.get_design_matrix()
        y = dataset.get_targets()
        if one_hot:
            y_, _ = compressed_one_hot(y_)
        indices = np.random.random_integers(0, X.shape[0] - 1)
        super(Bagged, self).__init__(X=X[indices], y=y[indices])
