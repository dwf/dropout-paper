from pylearn2.models.mlp import MLP


class HandicappedMLP(MLP):
    def __init__(self,
                 layers,
                 batch_size=None,
                 input_space=None,
                 nvis=None,
                 seed=None,
                 mask=None,
                 masked_input_layers=frozenset(),
                 default_input_scale=2.,
                 input_scales=None):
        self.mask = mask
        self.masked_input_layers = masked_input_layers
        if len(self.masked_input_layers) == 0 or mask is None:
            raise ValueError("You're not dropping anything.")
        self.default_input_scale = default_input_scale
        self.input_scales = input_scales if input_scales is not None else {}
        super(HandicappedMLP, self).__init__(layers, batch_size,
                                             input_space, nvis,
                                             seed)

    def fprop(self, state_below):
        return self.masked_fprop(state_below, self.mask,
                                 masked_input_layers=self.masked_input_layers,
                                 default_input_scale=self.default_input_scale,
                                 input_scales=self.input_scales)
