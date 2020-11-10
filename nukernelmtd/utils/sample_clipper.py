import numpy as np


def sample_clipper(samples, bounds):
    """A method to clip samples into range.
    Clip the samples according to the max and min of bounds.

    Args:
        samples (array-like): the shape of array is (n_samples,ndim)
        bounds (array-like): the shape of array is (n_dim, 2).
            If you want to clip 3-dimentional data, the bounds should be:
            bounds = np.array([[min, max],
                               [min, max],
                               [min, max]])

    Returns:
        numpy.array, int : Clipped samples and sample size after clipping.

    Examples:
        >>> samples = np.array([[1., 2.],[2., 5.],[3., 3.], [2., 4.]])
        >>> sample_clipper(samples,bounds=np.array([[2., 5.],[2., 4.]]))
        (array([[3., 3.],
               [2., 4.]]), 2)
    """
    if not isinstance(bounds, np.ndarray):
        bounds = np.array(bounds)
    if not isinstance(samples, np.ndarray):
        samples = np.array(samples)
    lists = np.all((samples >= bounds[:, 0]) & (samples <= bounds[:, 1]), axis=1)
    return samples[lists], lists.sum()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
