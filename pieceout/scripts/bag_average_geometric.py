#!/usr/bin/env python
import sys
import numpy as np
num_models = len(sys.argv[1:-1])
x = None
for m in sys.argv[1:-1]:
    assert "pre" in m
    print "processsing",m
    fprops = np.load(m)
    if x is None:
        x = fprops
    else:
        x += fprops
import pdb; pdb.set_trace()
x = x / float(num_models)
x = 1. / (1 + np.exp(-x))
print "Saving to", sys.argv[-1]
np.save(sys.argv[-1], x)
