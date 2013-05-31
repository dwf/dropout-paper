import sys
import theano
import numpy as np
from pylearn2.utils.serial import load
from pylearn2.datasets.mnist import MNIST
from pieceout.datasets.data_subsets import ClassificationSubtask
from pieceout.diamond import Diamond

model = load(sys.argv[1])

if sys.argv[1].startswith('mnist23'):
    data = ClassificationSubtask(dataset=MNIST('train', start=50000,
                                                stop=60000), classes=[2, 3])
elif sys.argv[1].startswith('diamond'):

    data = ClassificationSubtask(dataset=Diamond(1000, rng=3))


else:
    raise ValueError("WTF")
batch = model.get_input_space().make_batch_theano()
fprop = model.fprop(batch)
f = theano.function([batch], [fprop, fprop.owner.inputs[0]])
pre, post = f(data.get_design_matrix())
np.save(sys.argv[1] + '_pre.npy', pre.squeeze())
np.save(sys.argv[1] + '_post.npy', post.squeeze())

if sys.argv[1].startswith('diamond'):
    xmin = -1
    xmax = 1
    xres = 1000
    yres = 1000
    xx, yy = np.mgrid[xmin:xmax:xres * 1j, ymin:ymax:yres * 1j]
    coordinates = np.concatenate((xx.ravel()[:, np.newaxis],
                                yy.ravel()[:, np.newaxis]), axis=1)
    grid_pre, grid_post = f(coordinates)
    np.save(sys.argv[1] + '_grid_pre.npy', pre.squeeze())
    np.save(sys.argv[1] + '_grid_post.npy', post.squeeze())
