#!/usr/bin/env python

import numpy as np
from pylearn2.config import yaml_parse
from pylearn2.monitor import get_channel
from pylearn2.utils import serial
num_jobs = 40
import sys
name = sys.argv[1]
print name
for condition in ['drop', 'sgd', 'ens']:
    errs = np.zeros((num_jobs,))
    for job_num in xrange(num_jobs):
        failed = False
        try:
            model = serial.load('%s/' % name + condition + '/%s_' % name + condition + '.' + str(job_num) + '_best.pkl')
        except Exception:
            failed = True
            print condition,job_num,'failed'
        if failed:
            err = np.nan
        else:
            dataset_yaml_src = model.dataset_yaml_src
            dataset = yaml_parse.load(dataset_yaml_src)
            dataset = dataset.get_test_set()

            err =  get_channel(model, dataset, 'y_misclass', model.get_default_cost(), dataset.X.shape[0])
        errs[job_num] = err
    np.save(condition + '_test_err.npy', errs)
