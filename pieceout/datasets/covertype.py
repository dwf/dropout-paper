import os.path
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.string_utils import preprocess
from load_covertype import load_covertype


class CoverType(DenseDesignMatrix):
    def __init__(self, which_set, standardize_quantitative=True,
                 separate_types=False, prefix=None):
        if separate_types:
            raise NotImplementedError("This won't work as long as this "
                                      "is a subset of DenseDesignMatrix")
        self._separate_types = separate_types
        self._standardize_quantitative = standardize_quantitative
        self._prefix = prefix
        prefix = prefix if prefix is not None else "${PYLEARN2_DATA_PATH}"
        self._raw = load_covertype(
            preprocess(os.path.join(prefix, "covertype")),
            which_set=which_set,
            separate_types=self._separate_types,
            standardize_quantitative=self._standardize_quantitative
        )
        super(DenseDesignMatrix, self).__init__(
            X=self._raw[which_set]['features'],
            y=self._raw[which_set]['labels']
        )

    def get_test_set(self):
        return self.__class__('test', self._standardize_quantitative,
                              self._separate_types, self._prefix)
