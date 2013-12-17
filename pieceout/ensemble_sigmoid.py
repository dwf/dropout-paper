import numpy as np
import theano.tensor as T
from pylearn2.models.mlp import Sigmoid
from pylearn2.utils import block_gradient, sharedX


class MultiSigmoid(Sigmoid):
    def __init__(self, **kwargs):
        if 'mask_weights' not in kwargs:
            raise ValueError("mask_weights not specified")
        super(MultiSigmoid, self).__init__(self, **kwargs)
        self._gradient_mask = sharedX(np.ones(kwargs['dim']))

    def get_monitoring_channels_from_state(self, state, target=None):
        channels = super(MultiSigmoid, Sigmoid)
        for c in channels:
            if 'misclass' in c:
                del channels[c]
        z, = state.owner.inputs
        geo = T.nnet.sigmoid(z.mean(axis=1).dimshuffle(0, 'x'))
        misclass = T.neq(geo, target).mean()
        channels['misclass'] = misclass
        for i in range(self.dim):
            t = state[:, i:i + 1]
            channels['misclass_' + i] = T.neq(t, target).mean()
        return channels

    def cost(self, Y, Y_hat):
        # Pull out the argument to the sigmoid
        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op

        if not hasattr(op, 'scalar_op'):
            raise ValueError("Expected Y_hat to be generated by an Elemwise "
                             "op, got " + str(op) + " of type " +
                             str(type(op)))
        assert isinstance(op.scalar_op, T.nnet.sigm.ScalarSigmoid)
        z, = owner.inputs
        # Broadcasted multiplication with the gradient mask.
        z = (z * self._gradient_mask +
             block_gradient(z) * (1. - self._gradient_mask))
        # Geometric mean.
        z = z.mean(axis=1)

        term_1 = Y * T.nnet.softplus(-z)
        term_2 = (1 - Y) * T.nnet.softplus(z)

        total = term_1 + term_2
        assert total.ndim == 1

        return total
