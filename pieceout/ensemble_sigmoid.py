import logging
import numpy as np
import theano.tensor as T
import theano
from theano import config
from pylearn2.models.mlp import Sigmoid
from pylearn2.utils import block_gradient, sharedX
from pylearn2.space import VectorSpace


log = logging.getLogger(__name__)


class MultiSigmoid(Sigmoid):
    def __init__(self, monitor_individual=True, **kwargs):
        if 'mask_weights' not in kwargs:
            raise ValueError("mask_weights not specified")
        kwargs['monitor_style'] = 'classification'
        super(MultiSigmoid, self).__init__(**kwargs)
        self._monitor_individual = monitor_individual
        if self._monitor_individual:
            self._gradient_mask = sharedX(np.ones(kwargs['dim']),
                                          name='gradient_mask')

    def get_monitoring_channels_from_state(self, state, target=None):
        channels = super(MultiSigmoid,
                         self).get_monitoring_channels_from_state(state,
                                                                  target)
        for c in channels:
            if 'misclass' in c:
                del channels[c]
        z, = state.owner.inputs
        geo = T.nnet.sigmoid(z.mean(axis=1).dimshuffle(0, 'x'))
        geo_class = T.gt(geo, 0.5)
        misclass = T.cast(T.neq(geo_class, target), config.floatX).mean()
        channels['misclass'] = misclass
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
        if self._monitor_individual:
            z = (z * self._gradient_mask +
                 block_gradient(z) * (1. - self._gradient_mask))
        # Geometric mean.
        z = z.mean(axis=1)

        # Expecting binary targets.
        term_1 = Y[:, 0] * T.nnet.softplus(-z)
        term_2 = (1 - Y[:, 0]) * T.nnet.softplus(z)

        total = term_1 + term_2
        assert total.ndim == 1
        return total.mean()

    def get_output_space(self):
        # Hack to get pylearn2 to stop complaining about fprop returning
        # bigger stuff.
        return VectorSpace(1)

    def disable(self, expert):
        assert self._monitor_individual, "Not monitoring individual experts"
        v = self._gradient_mask.get_value()
        if v[expert] == 0:
            raise ValueError("already removed expert %d" % expert)
        v[expert] = 0
        self._gradient_mask.set_value(v)
        log.info("******* Disabling expert %d" % expert)

    def all_disabled(self):
        return np.all(self._gradient_mask.get_value(borrow=True) == 0)


class MultiSigmoidExtension(object):
    def __init__(self, layer, dataset, batch_size, timeout=100):
        self._layer = layer
        self._dataset = dataset
        self._batch_size = batch_size
        self._timeout = timeout
        self._best_h0_W = [None] * self._layer.dim
        self._best_h1_W = [None] * self._layer.dim
        self._best_y_W = [None] * self._layer.dim
        self._best_h0_b = [None] * self._layer.dim
        self._best_h1_b = [None] * self._layer.dim
        self._best_y_b = [None] * self._layer.dim

    def setup(self, model, *_):
        batch = model.get_input_space().make_batch_theano()
        target = model.get_output_space().make_batch_theano()
        self._ens_timeouts = (self._timeout *
                              np.ones(self._layer.dim,
                                      dtype='int32'))
        self._errors = sharedX(np.zeros(self._layer.dim,
                                        dtype=config.floatX),
                               name='errors')
        self._best_errors = np.inf * np.ones(self._layer.dim,
                                             dtype=config.floatX)
        btarget = T.addbroadcast(target, 1)
        btarget.name = "btarget"
        state = model.fprop(batch)
        ens_class = T.gt(state, 0.5)
        ens_class.name = "ens_class"
        batch_errors = T.cast(T.neq(ens_class, btarget),
                              config.floatX).sum(axis=0)
        batch_errors.name = "batch_errors"
        self._f = theano.function([batch, target],
                                  updates=[(self._errors,
                                            self._errors + batch_errors)])

    def on_monitor(self, model, *_):
        # Reset the shared variable.
        self._errors.set_value(np.zeros_like(
            self._errors.get_value(borrow=True),
            dtype=config.floatX)
        )
        # Accumulate errors.
        for b, t in self._dataset.iterator(mode='sequential', targets=True,
                                           batch_size=self._batch_size):
            self._f(b, t)
        # Pull them out of the shared variable.
        errs = self._errors.get_value(borrow=True)
        # Boolean mask of where the current error level is better than
        # the previous best.
        better = errs < self._best_errors
        # Reset the timeouts of those that are.
        self._ens_timeouts[better] = self._timeout
        log.info("%d subnetworks found new best. Mean improvement: %f" %
                 (sum(better), (self._best_errors - errs).mean()))
        # Update current best.
        self._best_errors[better] = errs[better]
        better_idxs = np.where(better)[0]
        h0_W = model.layers[0].get_weights()
        h0_b = model.layers[0].get_biases()
        h1_W = model.layers[1].get_weights()
        h1_b = model.layers[1].get_biases()
        y_W = model.layers[2].get_weights()
        y_b = model.layers[2].get_biases()

        for idx in better_idxs:
            NUM_UNITS = 10
            # HACK, hardcoding 10
            s = slice(idx * NUM_UNITS, (idx + 1) * NUM_UNITS)
            self._best_h0_W[idx] = h0_W[:, s]
            self._best_h0_b[idx] = h0_b[s]
            self._best_h1_W[idx] = h1_W[s, s]
            self._best_h1_b[idx] = h1_b[s]
            self._best_y_W[idx] = y_W[s, idx:idx + 1]
            self._best_y_b[idx] = y_b[s]

        # Decrease the timeout counter of those that aren't.
        self._ens_timeouts[(~better) & (self._ens_timeouts > 0)] -= 1
        # Disable the gradient flow for anyone who has reached 0.
        dis_idxs = np.where(self._ens_timeouts == 0)[0]
        for idx in dis_idxs:
            log.info("Restoring parameters and disabling subnetwork %d."
                     % idx)
            # Restore parameters of best subnet.
            h0_W[:, s] = self._best_h0_W[idx]
            h0_b[s] = self._best_h0_b[idx]
            h1_W[s, s] = self._best_h1_W[idx]
            h1_b[s] = self._best_h1_b[idx]
            y_W[s, idx:idx + 1] = self._best_y_W[idx]
            y_b[s] = self._best_y_b[idx]
            # Disable.
            self._layer.disable(idx)
        if len(dis_idxs) > 0:
            # Flush them to the shared variables in one swoop.
            log.info("Some subnetworks were reset... flushing parameters.")
            model.layers[0].set_weights(h0_W)
            model.layers[0].set_biases(h0_b)
            model.layers[1].set_weights(h1_W)
            model.layers[1].set_biases(h1_W)
            model.layers[2].set_weights(y_W)
            model.layers[2].set_biases(y_b)

        log.info("%d subnetworks were disabled." % len(dis_idxs))
        if self._layer.all_disabled():
            raise StopIteration()
