#!/usr/bin/env python
import sys
import numpy as np
num_models = len(sys.argv[1:-1])
x = None
for m in sys.argv[1:-1]:
    assert "post" in m
    fprops = np.load(m)
    if x is None:
        x = fprops
    else:
        x += fprops
x = x / float(num_models)
print "Saving to", sys.argv[-1]
np.save(sys.argv[-1], x)
