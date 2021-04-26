import numpy as np
import gradient_descent as gd


def rosenbrock(x, a=1, b=100):
    """
    Rosenbrock function to test the optimization method
    :param x: input
    :param a: parameter a
    :param b: parameter b
    :return: the value of rosenbrock function
    """
    return pow(a - x[0], 2) + b * pow(x[1] - pow(x[0], 2), 2)


def df_dx1(x): return 400 * pow(x[0], 3) - 400 * x[0] * x[1] + 2 * x[0] - 2


def df_dx2(x): return 200 * x[1] - 200 * pow(x[0], 2)


def fd(x): return np.array([df_dx1(x), df_dx2(x)])


def callback(objectiveValue, iterationNumber):
    print("Value " + str(objectiveValue) + " iteration " + str(iterationNumber))


if __name__ == "__main__":
    gd.gradient_descent(rosenbrock, fd, 100, callback)
