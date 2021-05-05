import numpy as np


def computeDerivatives(mode, f, x):
    if mode == "forward":
        return forwardComputing()
    if mode == "backward":
        return backwardComputing()
    if mode == "numerical":
        return numericalComputing(f, x, epsilon=0.01)
    return


def forwardComputing():
    # TODO:forward auto-diff
    return


def backwardComputing():
    # TODO:backward auto-diff
    return


def numericalComputing(f, x, epsilon):
    """
    For computing the derivatives using numerical methods
    :param f: function that needs to be computed
    :param x: point of f
    :param epsilon: small number
    :return: partial derivatives of f at point x
    """
    dims = len(x)
    ret = []
    for dim in range(dims):
        add = np.zeros(dims)
        add[dim] = epsilon
        ret.append(np.divide(f(x + add) - f(x - add), 2 * epsilon))
    return np.array(ret)
