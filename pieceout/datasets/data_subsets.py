import numpy as np

from pylearn2.utils.one_hot import compressed_one_hot
from pylearn2.datasets import DenseDesignMatrix


class ClassificationSubtask(DenseDesignMatrix):
    def __init__(self, dataset, classes, one_hot=True,
                 simplify_binary=True):
        self.args = locals()
        X = dataset.get_design_matrix()
        y = dataset.get_targets()
        assert set(classes).issubset(set(y.squeeze()))
        mask = (y.reshape((-1, 1)) == np.asarray(classes)).any(axis=1)
        X_ = X[mask]
        y_ = y[mask]
        if one_hot:
            y_, _ = compressed_one_hot(y_)
        else:
            for i, c in enumerate(classes):
                y_[y_ == c] = i
        super(ClassificationSubtask, self).__init__(X=X_, y=y_)

    def get_test_set(self):
        dataset = self.args['dataset'].get_test_set()
        return ClassificationSubtask(dataset, self.args['classes'],
                self.args['one_hot'], self.args['simplify_binary'])


class Bagged(DenseDesignMatrix):
    def __init__(self, dataset, seed):
        X = dataset.get_design_matrix()
        y = dataset.get_targets()
        rng = np.random.RandomState(seed)
        indices = rng.random_integers(0, X.shape[0] - 1, size=X.shape[0])
        super(Bagged, self).__init__(X=X[indices], y=y[indices])
