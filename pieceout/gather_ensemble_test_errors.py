import argparse
import numpy as np
import theano
import theano.tensor as T
#from pylearn2.utils.serial import load
import cPickle
from pylearn2.config import yaml_parse
from pylearn2.models.mlp import sampled_dropout_average

fname = "/data/lisatmp/goodfeli/random_search_rectifier_mnist/exp/22/job_best.pkl"


def make_error_fn(inp, out, true_label):
    return theano.function([inp, true_label],
                           T.neq(T.argmax(out, axis=1),
                                 T.argmax(true_label, axis=1)).sum())


def measure_test_error(fn, X, y, batch_size):
    errors = 0
    for i in xrange(0, X.shape[0], batch_size):
        Xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        errors += fn(Xb, yb)
    return errors / float(X.shape[0])


def make_parser():
    parser = argparse.ArgumentParser(
        description="Measure Monte-Carlo averages of dropout nets."
    )
    parser.add_argument('model', type=argparse.FileType('r'))
    parser.add_argument('-L', '--low-samples', type=int, default=2)
    parser.add_argument('-H', '--high-samples', type=int, default=121)
    parser.add_argument('-S', '--step-samples', type=int, default=2)
    parser.add_argument('-B', '--batch-size', type=int, default=500)
    parser.add_argument('-R', '--repeats', type=int, default=10)
    parser.add_argument('-P', '--per-example', action="store_const",
                        const=True, default=False)
    parser.add_argument('outfile', type=argparse.FileType('w'))
    parser.add_argument('-s', '--h0-scale', type=float, default=1.)
    parser.add_argument('-p', '--h0-prob', type=float, default=0.8)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    model = cPickle.load(args.model)
    src = model.dataset_yaml_src
    test = yaml_parse.load(src)
    test = test.get_test_set()
    assert test.X.shape[0] == 10000
    test.X = test.X.astype('float32')
    test.y = test.y.astype('float32')
    X = test.X
    y = test.y
    Xb = model.get_input_space().make_batch_theano()
    Xb.name = 'Xb'
    yb = model.get_output_space().make_batch_theano()
    yb.name = 'yb'
    # W/2 network
    fn = make_error_fn(Xb, model.fprop(Xb), yb)
    mf_test_error = measure_test_error(fn, X, y, batch_size=args.batch_size)
    print "Test error: %f" % mf_test_error
    num_masks = range(args.low_samples, args.high_samples, args.step_samples)
    results = np.empty((args.repeats, len(num_masks)), dtype='float64')
    for i, n_masks in enumerate(num_masks):
        print "Gathering results for n_masks = %d..." % n_masks
        out = sampled_dropout_average(model, Xb, n_masks,
                                      per_example=args.per_example,
                                      input_include_probs={'h0': args.h0_prob},
                                      input_scales={'h0': args.h0_scale})
        f = make_error_fn(Xb, out, yb)
        for rep in xrange(args.repeats):
            print "Repeat %d" % (rep + 1)
            results[rep, i] = measure_test_error(f, X, y,
                                                 batch_size=args.batch_size)
        print "Done."
    np.save(args.outfile, results)


if __name__ == "__main__":
    main()
