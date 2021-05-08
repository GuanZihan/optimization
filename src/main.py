import numpy as np
import gradient_descent as gd
import newton_method as nt
import trust_region as tr
import derivatives
import matplotlib.pyplot as plt


def rosenbrock(x, a=1, b=100):
    """
    Rosenbrock function to test the optimization method
    :param x: input
    :param a: parameter a
    :param b: parameter b
    :return: the value of rosenbrock function
    """
    return np.power(a - x[0], 2) + b * np.power(x[1] - np.power(x[0], 2), 2)


def callback(objectiveValue, currentSolution, iterationNumber):
    """
    callback function
    called when x updates in the optimization methods
    :param objectiveValue: objective function
    :param currentSolution: current solution x
    :param iterationNumber: current iteration number
    :return:
    """
    ax.scatter(currentSolution[0], currentSolution[1], objectiveValue, color='blue')
    print("Value " + str(objectiveValue) + " current solution " + str(currentSolution) + " iteration " + str(
        iterationNumber))


def gradient(x):
    """
    gradient function of objective function at point x
    :param x: input point
    :return:
    """
    return derivatives.computeDerivatives("numerical", rosenbrock, x)


def hessian(x):
    """
    hessian matrix of objective function at point x
    :param x: input point
    :return:
    """
    return derivatives.computeDerivatives("numerical_hessian", rosenbrock, x)


def plot_figure(function, XRange, YRange):
    """
    plot function in 3-d figure
    :param function:
    :param XRange:
    :param YRange:
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(XRange[0], XRange[1], 0.25)
    Y = np.arange(YRange[0], YRange[1], 0.25)

    X, Y = np.meshgrid(X, Y)
    Z_ = []
    for row in range(len(X)):
        temp = []
        for col in range(len(X)):
            temp.append(rosenbrock(np.array([X[row][col], Y[row][col]])))
        Z_.append(temp)

    Z = np.array(Z_)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="rainbow", alpha=0.7)
    return ax
def df_dx1(x): return 400*np.math.pow(x[0], 3) - 400*x[0]*x[1] + 2*x[0] - 2
def df_dx2(x): return 200*x[1] - 200*np.math.pow(x[0], 2)
def fd(x): return np.array([ df_dx1(x), df_dx2(x) ])

if __name__ == "__main__":
    ax = plot_figure(rosenbrock, [-4, 4], [-4, 4])
    # gd.gradient_descent(rosenbrock, gradient, 1000, callback)
    # nt.newton_method(rosenbrock, gradient, hessian, 1000, callback)
    tr.trust_region(rosenbrock, gradient, hessian, 100, 0.06, callback)
    plt.show()