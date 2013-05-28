import pylearn2.models.maxout
import pylearn2.models.mlp
import pylearn2.space


def replace_layers(mlp):
    for i, layer in enumerate(list(mlp.layers)):
        if isinstance(layer, pylearn2.models.maxout.Maxout):
            dim = layer.get_biases().shape[0]
            new_layer = pylearn2.models.mlp.Softplus(dim, layer.layer_name,
                                                     irange=0.0)
            new_layer.set_mlp(mlp)
            new_layer.set_input_space(layer.get_input_space())
            new_layer.set_biases(layer.get_biases())
            new_layer.set_weights(layer.get_weights())
            mlp.layers[i] = new_layer
    return mlp
