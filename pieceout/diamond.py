import numpy as np
from math import sqrt
from pylearn2.datasets import DenseDesignMatrix


def _nested_task(x, y, flip_horizontal=False, flip_vertical=False):
    z = np.zeros(x.shape, dtype='uint8')
    z[(x > 0) & (y > 0) & (y > -x + 2 - (sqrt(0.75)))] = 1
    z[(x < 0) & (y > 0) & (y > x + (2 - sqrt(0.75)))] = 1
    z[(x < 0) & (y < 0) & (y < -x - (2 - sqrt(0.75)))] = 1
    z[(x > 0) & (y < 0) & (y < x - (2 - sqrt(0.75)))] = 1
    # Inner diamond
    z[(x > 0) & (y > 0) & (y < -x + 0.5)] = 1.
    z[(x > 0) & (y < 0) & (y > x - 0.5)] = 1.
    z[(x < 0) & (y < 0) & (y > -x - 0.5)] = 1.
    z[(x < 0) & (y > 0) & (y < x + 0.5)] = 1.
    if flip_horizontal:
        z[x > 0] = 1 - z[x > 0]
    if flip_vertical:
        z[y > 0] = 1 - z[y > 0]
    return z


def _basic_task(x, y, flip_horizontal=False, flip_vertical=False):
    z = np.zeros(x.shape, dtype='uint8')
    # Inner diamond
    z[(x > 0) & (y > 0) & (y < -x + 1)] = 1.
    z[(x > 0) & (y < 0) & (y > x - 1)] = 1.
    z[(x < 0) & (y < 0) & (y > -x - 1)] = 1.
    z[(x < 0) & (y > 0) & (y < x + 1)] = 1.
    if flip_horizontal:
        z[x > 0] = 1 - z[x > 0]
    if flip_vertical:
        z[y > 0] = 1 - z[y > 0]
    return z


class SyntheticDiamond(DenseDesignMatrix):
    """

    Parameters
    ----------
    num_examples : int
        The number of examples to generate.

    two_targets : bool
        Encode the target as a length 2 one-hot vector.

    rng : RandomState or seed
        A random number generator or a seed used to construct it.

    flip_horizontal : bool, optional

    flip_vertical : bool, optional
    """
    _default_seed = (2013, 05, 17)

    def __init__(self, num_examples, two_targets=False, rng=(2013, 05, 22),
                 flip_horizontal=False, flip_vertical=False):
        if not hasattr(rng, 'uniform'):
            rng = np.random.RandomState(rng)
        X = rng.uniform(-1, 1, size=(num_examples, 2))
        y = self._label_fn(X[:, 0], X[:, 1]).reshape((-1, 1))
        if two_targets:
            y_hat = np.zeros(num_examples, dtype='uint8')
            y_hat.flat[np.arange(0, num_examples, 2) + y_hat] = 1
            y = y_hat
        ## FUCK THIS NOISE
        #X += 1
        #X /= 2
        super(SyntheticDiamond, self).__init__(X=X, y=y)


class Diamond(SyntheticDiamond):
    _label_fn = lambda self, *args, **kwargs: _basic_task(*args, **kwargs)

    def get_test_set(self):
        return Diamond(1000, rng=3)

class NestedDiamond(SyntheticDiamond):
    _label_fn = lambda self, *args, **kwargs: _nested_task(*args, **kwargs)
