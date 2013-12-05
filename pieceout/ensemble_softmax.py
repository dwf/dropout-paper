import theano.tensor as T
from collections import OrderedDict
from pylearn2.models.mlp import Softmax


class MultiSoftmax(Softmax):

    def __init__(self, n_classes, n_replicas, layer_name, irange=None,
                 istdev=None, sparse_init=None, W_lr_scale=None,
                 b_lr_scale=None, max_row_norm=None, no_affine=False,
                 max_col_norm=None, init_bias_target_marginals=None):
        """
        """
        d = dict(locals())
        del d['n_classes']
        del d['n_replicas']
        super(Softmax, self).__init__(n_classes * n_replicas, **d)
        self._n_replicas = n_replicas

    def get_monitoring_channels_from_state(self, state, target=None):
        per_softmax = state.shape[1] / self._n_replicas
        rval = OrderedDict()
        for softmax in xrange(self._n_replicas):
            s = state[:, softmax * per_softmax:(softmax + 1) * per_softmax]
            t = super(MultiSoftmax,
                      self).get_monitoring_channels_from_state(s, target)
            rval.update(('softmax' + softmax + '_' + key, val)
                        for key, val in t.iteritems())
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
        # TODO: implement this, using gradient-blocking to mask out
        # terms from things we've terminated training for.
        raise NotImplementedError()
