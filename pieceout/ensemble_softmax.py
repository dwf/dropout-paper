from collections import OrderedDict
import numpy as np
import pieceout
from pylearn2.utils import block_gradient, sharedX
from pylearn2.models.mlp import Softmax
import theano
import theano.tensor as T
from .extract_arg import extract_op_argument


class MultiSoftmax(Softmax):
    def __init__(self, n_classes, n_replicas, layer_name, irange=None,
                 istdev=None, sparse_init=None, W_lr_scale=None,
                 b_lr_scale=None, max_row_norm=None, no_affine=False,
                 max_col_norm=None, init_bias_target_marginals=None,
                 arithmetic_cost=False):
        """
        """
        d = dict(locals())
        del d['n_classes']
        del d['n_replicas']
        del d['arithmetic_cost']
        self._cost_mask = theano.shared(np.ones(n_replicas,
                                                dtype=theano.config.floatX))
        super(Softmax, self).__init__(n_classes * n_replicas, **d)
        self._n_classes = n_classes
        self._n_replicas = n_replicas

    def arithmetic_mean(self, state):
        reshaped = state.reshape((state.shape[0], self._n_replicas,
                                  state.shape[1] / self._n_replicas))
        broadcasted_mask = self._grad_mask.dimshuffle('x', 0, 'x')
        unblocked = reshaped * broadcasted_mask
        blocked = block_gradient(reshaped) * (np.float32(1) - broadcasted_mask)
        return (unblocked + blocked).mean(axis=1)

    def geometric_mean(self, state):
        pre = extract_op_argument(state, recurse=2)
        rsh = pre.reshape((pre.shape[0], self._n_replicas, pre.shape[1] /
                          self._n_replicas))
        broadcasted_mask = self._grad_mask.dimshuffle('x', 0, 'x')
        unblocked = rsh * broadcasted_mask
        blocked = block_gradient(rsh) * (np.float32(1) - broadcasted_mask)
        geo = T.nnet.softmax((unblocked + blocked).mean(axis=1))
        return geo

    def get_monitoring_channels_from_state(self, state, target=None):
        per_softmax = state.shape[1] / self._n_replicas
        rval = OrderedDict()
        for softmax in xrange(self._n_replicas):
            s = state[:, softmax * per_softmax:(softmax + 1) * per_softmax]
            t = super(MultiSoftmax,
                      self).get_monitoring_channels_from_state(s, target)
            rval.update(('softmax_' + softmax + '_' + key, val)
                        for key, val in t.iteritems() if 'misclass' in key)
        # Arithmetic mean.
        a = self.arithmetic_mean(state)
        at = super(MultiSoftmax,
                   self).get_monitoring_channels_from_state(a, target)
        rval.update(('arithmetic_mean_' + key, val)
                    for key, val in at.iteritems())
        # Geometric mean.
        geo = self.geometric_mean(state)
        gt = super(MultiSoftmax,
                   self).get_monitoring_channels_from_state(geo, target)
        rval.update(('geometric_mean_' + key, val)
                    for key, val in gt.iteritems())
        return rval

    def fprop(self, state_below):
        self.input_space.validate(state_below)
        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)
        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            Z = state_below
        else:
            assert self.W.ndim == 2
            b = self.b
            Z = T.dot(state_below, self.W) + b
        inter = Z.reshape((Z.shape[0] * self._n_replicas,
                           Z.shape[1] / self._n_replicas))
        soft = T.nnet.softmax(inter)
        rval = soft.reshape(Z.shape)
        return rval

    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.
        """
        if self._arithmetic_cost:
            return super(MultiSoftmax,
                         self).cost(Y, self.arithmetic_mean(Y_hat))
        else:
            return super(MultiSoftmax,
                         self).cost(Y, self.geometric_mean(Y_hat))

    def censor_updates(self, updates):
        updates = super(MultiSoftmax, self).censor_updates(updates)
        updates[self.W] = updates[self.W] * sharedX(mask)
        return updates

    def set_input_space(self, space):
        super(MultiSoftmax, self).set_input_space(space)
        # Masks out a block diagonal so that each softmax takes input
        # only from all_input_dim / n_replicas features in the layer
        # below.
        all_input_dim = self.W.get_value(borrow=True).shape[0]
        assert all_input_dim % self._n_replicas == 0
        input_dim = all_input_dim / self._n_replicas
        self.mask = pieceout.block_diagonal_mask(input_dim, self._n_classes,
                                                 self._n_replicas)
        self.W.set_value(self.W.get_value(borrow=True) * self.mask)
