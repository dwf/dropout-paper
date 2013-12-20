#!/usr/bin/env python
import cPickle
import argparse
import numpy as np
from pylearn2.config.yaml_parse import load as yload
from pylearn2.utils.timing import log_timing
from pieceout.fprop_ensemble import compare_ensemble
import pandas as pd

import logging
log = logging.getLogger()
logging.basicConfig()
log.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Collect statistics.')
    parser.add_argument('infile', nargs='+', type=argparse.FileType('r'),
                        help="The pickle files to read.")
    parser.add_argument('-O', '--output', type=argparse.FileType('w'),
                        help="Output CSV file to write.")
    args = parser.parse_args()
    names = [(a.name, a) for a in args.infile]
    indices = [b[0] for b in names]
    columns = ['weight_scaling_error', 'geometric_error',
               'arithmetic_error']
    df = pd.DataFrame(index=indices, columns=columns)
    try:
        dataset = None
        for i, (name, model_handle) in enumerate(names):
            with log_timing(log, "Processing %s [%d / %d]" %
                            (name, i + 1, len(names))):
                with model_handle as f:
                    model = cPickle.load(f)
                if dataset is None:
                    with log_timing(log, "Loading test set",
                                    final_msg="Loaded."):
                        d = None
                        # HACK HACK HACK
                        for k in model.monitor._datasets:
                            if 'valid' in k or '50000' in k:
                                d = k
                                break
                        if d is None:
                            log.warning("No validation set found, using "
                                        "first dataset in monitor.")
                            d = model.monitor._datasets[0]
                        dataset = yload(d).get_test_set()
                d = compare_ensemble(model, dataset, input_scales={'h1': 2.,
                                                                   'y': 2.})
                df['weight_scaling_error'][name] = d['weight_scaling_error']
                df['geometric_error'][name] = d['geometric_error']
                df['arithmetic_error'][name] = d['arithmetic_error']
                np.save(model_handle.name + '.sca.npy',
                        d['weight_scaling_output'])
                np.save(model_handle.name + '.geo.npy', d['geometric_output'])
                np.save(model_handle.name + '.ari.npy', d['arithmetic_output'])
    finally:
        df.to_csv(args.output)


if __name__ == "__main__":
    main()
