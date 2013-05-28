import numpy as np
import matplotlib.pyplot as plt


def plot_2d_decision_surface(callable_obj, xmin=-1, xmax=1, ymin=-1, ymax=1,
                             xres=500, yres=500, **kwargs):
    """
    Parameters
    ----------
    callable_obj : callable
        A callable object such as a Python function or Theano
        function that hands back one class label per point.

    xmin, xmax, ymin, ymax : floats, optional
        The extents of the grid of points. Defaults to -1, 1, -1, 1.

    xres, yres : floats, optional
        The resolution of the grid along the x and y dimensions.

    Returns
    -------
    image : matplotlib.image.AxisImage
        The return value of an `imshow()` call.
    """
    xx, yy = np.mgrid[xmin:xmax:xres * 1j, ymin:ymax:yres * 1j]
    coordinates = np.concatenate((xx.ravel()[:, np.newaxis],
                                  yy.ravel()[:, np.newaxis]), axis=1)
    labels = callable_obj(coordinates).reshape((yres, xres))
    return plt.imshow(labels, extent=[xmin, xmax, ymin, ymax],
                      **kwargs)
