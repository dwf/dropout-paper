import numpy
from theano import config


def block_diagonal_mask(block_rows, block_cols, repeat):
    mask = numpy.zeros((repeat * block_rows, repeat * block_cols),
                       dtype=config.floatX)
    for i in xrange(repeat):
        mask[i * block_rows:(i + 1) * block_rows,
             i * block_cols:(i + 1) * block_cols] = 1.
    return mask


def test_block_diagonal_mask():
    expected = numpy.array([[1., 1., 0., 0., 0., 0.],
                            [0., 0., 1., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 1.]], dtype=config.floatX)
    numpy.testing.assert_allclose(expected, block_diagonal_mask(1, 2, 3))

    expected = numpy.array([[1., 1., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0., 0.],
                            [0., 0., 1., 1., 0., 0.],
                            [0., 0., 1., 1., 0., 0.],
                            [0., 0., 1., 1., 0., 0.],
                            [0., 0., 1., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 1.],
                            [0., 0., 0., 0., 1., 1.],
                            [0., 0., 0., 0., 1., 1.],
                            [0., 0., 0., 0., 1., 1.]], dtype=config.floatX)
    numpy.testing.assert_allclose(expected, block_diagonal_mask(4, 2, 3))
