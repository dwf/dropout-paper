import hashlib
import os.path
import numpy as np
import theano


# The first 10 columns are measurements that should be treated as
# real-valued.
QUANTITATIVE_COLS = 10
NUM_ITEMS = {'train': 11340, 'valid': 3780, 'test': 565892}
SLICES = {'train': slice(0, NUM_ITEMS['train']),
          'valid': slice(NUM_ITEMS['train'],
                         NUM_ITEMS['train'] + NUM_ITEMS['valid']),
          'test': slice(NUM_ITEMS['train'] + NUM_ITEMS['valid'],
                        NUM_ITEMS['train'] + NUM_ITEMS['valid'] +
                        NUM_ITEMS['test'])}

FILENAME_PREFIX = 'covtype.data'
SHA1 = {'gz': 'e757fc472346f7d623b0a410dc856afabedec310',
        'data': 'da39a3ee5e6b4b0d3255bfef95601890afd80709'}


def verify_sha1(fname, expected_digest, block_size=128):
    sha1 = hashlib.sha1()
    with open(fname, 'rb') as f:
        while True:
            data = f.read(block_size)
            if not data:
                break
            sha1.update(data)
    actual_digest = sha1.hexdigest()
    if actual_digest != expected_digest:
        raise ValueError('expected SHA-1 digest of %s for %s, got %s' %
                         (expected_digest, fname, actual_digest))


def prefer_gzipped(path, filename):
    full_name = os.path.join(path, filename)
    if os.path.isfile(full_name + '.gz'):
        fname = full_name + '.gz'
    elif os.path.isfile(full_name):
        fname = full_name
    else:
        raise IOError("Couldn't load %s or %s.gz from %s" %
                      (FILENAME_PREFIX, FILENAME_PREFIX + '.gz', path))
    return fname


def validate_items(items, valid):
    if not all(w in valid for w in items):
        w = [s for s in items if s not in valid]
        raise ValueError("%s in 'which' argument not one of valid %s"
                         (str(w), str(tuple(valid))))


def load_covertype(path, which_set=('train', 'valid'), separate_types=False,
                   standardize_quantitative=False, dtype=None):
    # Use floatX for quantitative variables if nothing is specified.
    dtype = theano.config.floatX if dtype is None else dtype
    which = (which,) if isinstance(which, basestring) else which
    # Make sure every set being requested is one of 'train', 'valid', 'test'.
    valid_sets = frozenset(['train', 'valid', 'test'])
    validate_items(which, valid_sets)
    fname = prefer_gzipped(path, FILENAME_PREFIX)
    # TODO: inefficient to load the whole file, but loadtxt
    all_data = np.loadtxt(fname, delimiter=',', dtype='int16')
    # Calculate standardization constants based on training set only.
    # Doing it per-set would be, in a way, cheating.
    if standardize_quantitative:
        quant_mean = all_data[SLICES['train'], :QUANTITATIVE_COLS].mean(axis=0)
        quant_std = all_data[SLICES['train'], :QUANTITATIVE_COLS].std(axis=0)
    # Accumulate a dictionary of dictionaries.
    results = {}
    for which_set in which:
        results[which_set] = d = {}
        # If the caller requested that qualitative and quantitative
        # be separated, put two items in the dictionary.
        if separate_types:
            d['qualitative'] = all_data[SLICES[which_set],
                                        QUANTITATIVE_COLS:-1]
            d['quantitative'] = all_data[SLICES[which_set],
                                         :QUANTITATIVE_COLS].astype(dtype)
        # Otherwise, just one, call it "features".
        else:
            d['features'] = all_data[SLICES[which_set], :-1].astype(dtype)
        # Standardize only the quantitative variables in either case.
        if standardize_quantitative and separate_types:
            d['quantitative'] -= quant_mean
            d['quantitative'] /= quant_std
        elif standardize_quantitative:
            d['features'][:, :QUANTITATIVE_COLS] -= quant_mean
            d['features'][:, :QUANTITATIVE_COLS] /= quant_std
        # Add the labels, regardless of any of the above.
        d['labels'] = all_data[SLICES[which_set], -1]
    return results
