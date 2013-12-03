from collections import OrderedDict
from itertools import izip
import theano
from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace


def extract_op_argument(obj):
    assert len(obj.owner.inputs) == 1
    return obj.owner.inputs[0]


class DropoutBoosting(Cost):
    supervised = True

    def __init__(self, same_mask=True, negative_scale=1.,
                 num_negative_samples=0, default_input_include_prob=0.5,
                 input_include_probs=None, default_input_scale=2.,
                 input_scales=None):
        """
        Parameters
        ----------
        same_mask : bool, optional
            Use the same mask for both the positive phase
            and negative phase. Default is `True`.

        negative_scale : float
            Weight the negative phase gradient by this factor.

        num_negative_samples : int, optional
            If 0, use the deterministic approximate ensemble
            prediction to generate the negative targets. If greater
            than 0, use that many sampled masks, geometrically
            averaged together.

        default_input_include_prob : float, optional
        input_include_probs : dict, optional
        input_include_probs : float, optional
        input_scales : dict, optional
            Passed along to `dropout_fprop`.
        """
        self._same_mask = same_mask
        self._negative_scale = negative_scale
        assert num_negative_samples >= 0
        self._num_negative_samples = num_negative_samples
        self._default_input_include_prob = default_input_include_prob
        self._input_include_probs = input_include_probs
        self._default_input_scale = default_input_scale
        self._input_scales = input_scales

    def _gradients_from_prediction(self, model, Y, Yhat):
        cost = model.cost(Y, Yhat)
        params = list(model.get_params())
        grads = theano.tensor.grad(cost, params, disconnected_inputs='ignore',
                                   consider_constant=[Y])
        gradients = OrderedDict(izip(params, grads))
        return gradients

    def expr(self, model, data, **kwargs):
        self.get_data_specs(model)[0].validate(data)
        return model.cost_from_X(data)

    def get_gradients(self, model, data):
        X, Y = data
        # Step one: forward prop with the dropout mask.
        pos_y = model.dropout_fprop(
            X,
            default_input_include_prob=self._default_input_include_prob,
            input_include_probs=self._input_include_probs,
            default_input_scale=self._default_input_scale,
            input_scales=self._input_scales
        )

        # Step two: get the gradient with respect to this dropout mask fprop,
        # and the real labels.
        pg_y = self._gradients_from_prediction(model, Y, pos_y)

        # Get the "negative phase targets", the ensemble predictions.
        if self._num_negative_samples:
            # Forward prop with one dropout mask.
            fprop = model.dropout_fprop(
                X,
                default_input_include_prob=self._default_input_include_prob,
                input_include_probs=self._input_include_probs,
                default_input_scale=self._default_input_scale,
                input_scales=self._input_scales
            )
            # Average together one or more dropout masks for a sampling
            # approximation of the ensemble prediction.
            neg_pre = extract_op_argument(fprop)
            for i in xrange(self._num_negative_samples - 1):
                fprop = model.dropout_fprop(X)
                neg_pre += extract_op_argument(fprop)
            neg_pre /= self._num_negative_samples

            # Shove it through the softmax once again.
            neg_target = fprop.owner.op(neg_pre)
        else:
            # Otherwise use the deterministic approximate ensemble
            # prediction.
            neg_target = model.fprop(X)

        # Either use the same forward-prop graph or a different one
        # to comapre against the negative targets.
        if self._same_mask:
            neg_y = pos_y
        else:
            neg_y = model.dropout_fprop(
                X,
                default_input_include_prob=self._default_input_include_prob,
                input_include_probs=self._input_include_probs,
                default_input_scale=self._default_input_scale,
                input_scales=self._input_scales
            )

        # Get the negative phase gradients.
        ng_y = self._gradients_from_prediction(model, neg_target, neg_y)
        g_y = OrderedDict()

        # Get the differences between positive and negative phases.
        for key in pg_y:
            g_y[key] = pg_y[key] - self._negative_scale * ng_y[key]
        return g_y, OrderedDict()

    def get_data_specs(self, model):
        data = CompositeSpace([model.get_input_space(),
                               model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (data, sources)
