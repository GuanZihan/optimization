import numpy as np
import gradient_descent as gd
import derivatives


def rosenbrock(x, a=1, b=100):
    """
    Rosenbrock function to test the optimization method
    :param x: input
    :param a: parameter a
    :param b: parameter b
    :return: the value of rosenbrock function
    """
    return np.power(a - x[0], 2) + b * np.power(x[1] - np.power(x[0], 2), 2)


def callback(objectiveValue, iterationNumber):
    print("Value " + str(objectiveValue) + " iteration " + str(iterationNumber))


def fd(x):
    return derivatives.computeDerivatives("numerical", rosenbrock, x)


if __name__ == "__main__":
    gd.gradient_descent(rosenbrock, fd, 100, callback)
