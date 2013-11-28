__author__ = "Ian Goodfellow"

import warnings

from theano import tensor as T
from theano.printing import Print
from theano.sandbox import cuda
from theano.sandbox.rng_mrg import MRG_RandomStreams

from pylearn2.models.maxout import Maxout
from pylearn2.models.maxout import MaxoutConvC01B
from pylearn2.models.mlp import MLP
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from pylearn2.sandbox.cuda_convnet import check_cuda
from pylearn2.utils import safe_zip

class PieceChangeMonitoringMLP(MLP):

    def get_monitoring_channels(self, data):
        """
        data is a flat tuple, and can contain features, targets, or both
        """
        rval = super(PieceChangeMonitoringMLP, self).get_monitoring_channels(data)
        X, Y = data
        state = X

        theano_rng = MRG_RandomStreams(self.rng.randint(2 ** 15))

        assert not isinstance(state, tuple)
        piece_ids_0 = self.piece_id(state, theano_rng)
        # piece_ids_0 = Print('piece_ids_0[0]')(piece_ids_0[0])
        piece_ids_1 = self.piece_id(state, theano_rng)
        assert len(piece_ids_0) == 2 # rm

        piece_changes = T.cast(sum([T.neq(ids_0, ids_1).sum() for ids_0, ids_1 in safe_zip(piece_ids_0, piece_ids_1)]), 'float32')
        possible_changes = T.cast(sum([ids_0.size for ids_0 in piece_ids_0]), 'float32')
        rval['piece_change_rate'] = piece_changes / possible_changes

        return rval

    def piece_id(self, state_below, theano_rng, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):
        """
        state_below: The input to the MLP

        Returns the numerical index of the winning piece in each piecewise linear layer of the MLP,
        when applying dropout to the input and intermediate layers.
        Each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.

        per_example : bool, optional
            Sample a different mask value for every example in
            a batch. Default is `True`. If `False`, sample one
            mask per mini-batch.
        """

        warnings.warn("dropout doesn't use fixed_var_descr so it won't work with "
                "algorithms that make more than one theano function call per batch,"
                " such as BGD. Implementing fixed_var descr could increase the memory"
                " usage though.")

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        pieces = []

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )
            if hasattr(layer, 'piece_prop'):
                state_below, piece = layer.piece_prop(state_below)
                assert not isinstance(state_below, tuple)
                pieces.append(piece)
            else:
                break

        return pieces

class PieceAwareMaxout(Maxout):

    def piece_prop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p = T.zeros_like(z)
        else:
            p = None

        last_start = self.detector_layer_dim  - self.pool_size
        pieces = None
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p = cur
                pieces = T.zeros_like(p)
            else:
                p = T.maximum(cur, p)
                mask = T.eq(p, cur)
                pieces = (1 - mask) * pieces + mask * i
                # pieces = Print('pieces')(pieces)

        p.name = self.layer_name + '_p_'

        assert not isinstance(p, tuple)

        return p, pieces

class PieceAwareMaxoutConvC01B(MaxoutConvC01B):

    def piece_prop(self, state_below):
        """
        Note: this only reports pieces in terms of which channel wins, not
        which spatial location wins. Depending on the input size, it may report
        a piece map for either the pre-spatial pooling or the post-spatial pooling
        tensor.
        """
        check_cuda(str(type(self)))

        self.input_space.validate(state_below)

        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        if self.input_normalization:
            state_below = self.input_normalization(state_below)

        # Alex's code requires # input channels to be <= 3 or a multiple of 4
        # so we add dummy channels if necessary
        if not hasattr(self, 'dummy_channels'):
            self.dummy_channels = 0
        if self.dummy_channels > 0:
            state_below = T.concatenate((state_below,
                                         T.zeros_like(state_below[0:self.dummy_channels, :, :, :])),
                                        axis=0)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')


        z = z + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        self.detector_space.validate(z)

        assert self.detector_space.num_channels % 16 == 0

        if self.output_space.num_channels % 16 == 0:
            # alex's max pool op only works when the number of channels
            # is divisible by 16. we can only do the cross-channel pooling
            # first if the cross-channel pooling preserves that property
            piece = None
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                        piece = T.zeros_like(t)
                    else:
                        s = T.maximum(s, t)
                        mask = T.eq(s, t)
                        piece = mask * i + (1 - mask) * piece
                z = s

            if self.detector_normalization:
                z = self.detector_normalization(z)

            p = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                        "layer because the detector layer never exists as a "
                        "stage of processing in this implementation.")
            z = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
            if self.num_pieces != 1:
                s = None
                piece = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                        piece = T.zeros_like(t)
                    else:
                        s = T.maximum(s, t)
                        mask = T.eq(s, t)
                        piece = mask * i + (1- mask) * piece
                z = s
            p = z


        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p, piece
