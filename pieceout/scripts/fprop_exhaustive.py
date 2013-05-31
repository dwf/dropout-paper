#!/usr/bin/env python
import sys
import numpy as np

from pylearn2.datasets.mnist import MNIST
from pieceout.datasets.data_subsets import ClassificationSubtask
from pieceout.datasets.diamond import Diamond
from pieceout.fprop_ensemble import fprop_ensemble, compare_ensemble
from pylearn2.utils.serial import load

if sys.argv[1].startswith('mnist23'):
    if sys.argv[2] == 'valid':
        data = ClassificationSubtask(dataset=MNIST('train', start=50000, stop=60000), classes=[2, 3])
    elif sys.argv[2] == 'test':
        data = ClassificationSubtask(dataset=MNIST('train', start=50000, stop=60000), classes=[2, 3])
    else:
        raise ValueError('WTF')
elif sys.argv[1].startswith('diamond'):
    if sys.argv[2] == 'valid':
        data = Diamond(500, rng=2)
    elif sys.argv[2] == 'test':
        data = Diamond(1000, rng=3)
    else:
        raise ValueError('WTF')
else:
    raise ValueError("WTF")

model = load(sys.argv[1])
d = compare_ensemble(model, data, input_scales={'h1': 2., 'y': 2.})
print sys.argv[1], sys.argv[2], ":", 'ens', d['ensemble_pred_error'], 'w/2', d['approx_pred_error']
np.save(sys.argv[1] + '_' + sys.argv[2] + '_approx.npy', d['approx_prob'])
np.save(sys.argv[1] + '_' + sys.argv[2] + '_ensemble.npy', d['ensemble_prob'])
