import numpy as np
import pandas as pd


def transform_data(x):
    """data transform

    Args:
        x (array-like): transform input data into 2d-array of shape (N, n_feature).

    Raises:
        ValueError: [description]

    Returns:
        array-like : [description]

    Examples:
        >>> transform_data([1,2,3])
        array([[1.],
               [2.],
               [3.]])

        >>> transform_data([[1,2,3]])
        array([[1., 2., 3.]])

    """
    if isinstance(x, np.ndarray):
        if len(x.shape) == 1:
            return np.atleast_2d(x.astype(np.float64)).T
        else:
            return np.atleast_2d(x.astype(np.float64))
    elif isinstance(x, list):
        return transform_data(np.array(x))
    elif isinstance(x, pd.DataFrame):
        return transform_data(x.values)
    else:
        raise ValueError("Cannot convert to numpy.array")


if __name__ == '__main__':
    import doctest
    doctest.testmod()
