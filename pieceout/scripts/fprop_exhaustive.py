#!/usr/bin/env python
import sys
import numpy as np

from pylearn2.datasets.mnist import MNIST
from pieceout.datasets.data_subsets import ClassificationSubtask
from pieceout.datasets.diamond import Diamond
from pieceout.datasets.covertype import CoverType
from pieceout.fprop_ensemble import compare_ensemble
from pylearn2.utils.serial import load

if sys.argv[1].startswith('mnist23'):
    if sys.argv[2] == 'valid':
        data = ClassificationSubtask(dataset=MNIST('train', start=50000,
                                                   stop=60000), classes=[2, 3])
    elif sys.argv[2] == 'test':
        data = ClassificationSubtask(dataset=MNIST('train', start=50000,
                                                   stop=60000), classes=[2, 3])
    else:
        raise ValueError('WTF')
elif sys.argv[1].startswith('diamond'):
    if sys.argv[2] == 'valid':
        data = Diamond(500, rng=2)
    elif sys.argv[2] == 'test':
        data = Diamond(1000, rng=3)
    else:
        raise ValueError('WTF')
elif sys.argv[1].startswith('covertype'):
    if sys.argv[2] == 'valid':
        data = ClassificationSubtask(CoverType('valid', prefix='/RQexec/wardefar/data'), classes=[0, 1])
    elif sys.argv[2] == 'test':
        data = ClassificationSubtask(CoverType('test', prefix='/RQexec/wardefar/data'), classes=[0, 1])
    else:
        raise ValueError('WTF')
else:
    raise ValueError("WTF")

model = load(sys.argv[1])
d = compare_ensemble(model, data, input_scales={'h1': 2., 'y': 2.})
print sys.argv[1], sys.argv[2], ":", 'geo', d['ensemble_pred_error_geo'], 'ari', d['ensemble_pred_error_ari'], 'w/2', d['approx_pred_error']
np.save(sys.argv[1] + '_' + sys.argv[2] + '_approx.npy', d['approx_prob'])
np.save(sys.argv[1] + '_' + sys.argv[2] + '_geo.npy', d['ensemble_prob_geo'])
np.save(sys.argv[1] + '_' + sys.argv[2] + '_ari.npy', d['ensemble_prob_ari'])
np.save(sys.argv[1] + '_' + sys.argv[2] + '_approx_err.npy', d['approx_pred_error'])
np.save(sys.argv[1] + '_' + sys.argv[2] + '_geo_err.npy', d['ensemble_pred_error_geo'])
np.save(sys.argv[1] + '_' + sys.argv[2] + '_ari_err.npy', d['ensemble_pred_error_ari'])
