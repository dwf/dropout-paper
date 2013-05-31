import numpy as np
import theano
from pylearn2.utils.bit_strings import all_bit_strings


def fprop_ensemble(mlp, input_scales, arithmetic_mean=False, batch=None):
    """
    fprop a single example with all possible dropout masks.

    input_scales.keys() is the layers that get dropped.

    ONLY USEFUL WITH p = 0.5!
    """
    if batch is None:
        batch = mlp.get_input_space().make_batch_theano(batch_size=1)
    inputs_dim = mlp.get_total_input_dimension(input_scales.keys())
    masks = all_bit_strings(inputs_dim)
    state = batch
    for layer in mlp.layers:
        if layer.layer_name in input_scales:
            layer_input_dim = layer.get_input_space().get_total_dimension()
            these_masks = theano.tensor.as_tensor_variable(
                masks[:, :layer_input_dim])
            masks = masks[:, layer_input_dim:]
            state = layer.fprop(input_scales[layer.layer_name] *
                                state * these_masks)
        else:
            state = layer.fprop(state)
    assert masks.size == 0
    if arithmetic_mean:
        state = state.mean(axis=0)
    else:
        mean_pre = state.owner.inputs[0].mean(axis=0)
        state = state.owner.op(mean_pre)
    return batch, state


def compare_ensemble(mlp, dataset, input_scales):
    batch, state_geo = fprop_ensemble(mlp, input_scales, False)
    batch, state_ari = fprop_ensemble(mlp, input_scales, True, batch=batch)
    f = theano.function([batch], [state_geo, state_geo > 0.5,
                                  state_ari, state_ari > 0.5],
                        allow_input_downcast=True)
    ensemble_probs_geo = np.empty(dataset.get_design_matrix().shape[0],
                                  dtype='float32')
    ensemble_preds_geo = np.empty(dataset.get_design_matrix().shape[0],
                                  dtype='uint8')
    ensemble_probs_ari = np.empty(dataset.get_design_matrix().shape[0],
                                  dtype='float32')
    ensemble_preds_ari = np.empty(dataset.get_design_matrix().shape[0],
                                  dtype='uint8')
    for i, row in enumerate(dataset.get_design_matrix()):
        prob_geo, pred_geo, prob_ari, pred_ari = f(row.reshape((1, -1)))
        ensemble_probs_geo[i] = prob_geo
        ensemble_preds_geo[i] = pred_geo
        ensemble_probs_ari[i] = prob_ari
        ensemble_preds_ari[i] = pred_ari
    batch = mlp.get_input_space().make_batch_theano()
    g = theano.function([batch], [mlp.fprop(batch), mlp.fprop(batch) > 0.5],
                        allow_input_downcast=True)
    approx_prob, approx_pred = g(dataset.get_design_matrix())
    approx_prob = approx_prob.squeeze()
    approx_pred = approx_pred.squeeze()
    targets = dataset.get_targets().squeeze()

    results = {}
    results['approx_pred_error'] = (approx_pred != targets).mean()
    results['ensemble_pred_error_geo'] = (ensemble_preds_geo != targets).mean()
    results['ensemble_pred_error_ari'] = (ensemble_preds_ari != targets).mean()
    results['approx_prob'] = approx_prob
    results['ensemble_prob_geo'] = ensemble_probs_geo
    results['ensemble_prob_ari'] = ensemble_probs_ari
    return results
