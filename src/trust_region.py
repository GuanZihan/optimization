import numpy as np


def model_function(x, objective, gradient, hessian,  p):
    return objective(x) + np.matmul(np.transpose(gradient(x)), p) + 1/2 * np.matmul(np.matmul(np.transpose(p), hessian(x)), p)


def cauchy_point(x, gradient, hessian, delta):
    alpha = np.linalg.norm(gradient(x), 2)
    if delta > alpha:
        return np.multiply(np.linalg.norm(gradient(x), 2), gradient(x)) / np.matmul(np.matmul(np.transpose(gradient(x)), hessian(x)), gradient(x))
    else:
        return - np.multiply(delta / np.linalg.norm(gradient(x), 2), gradient(x))


def cauchy_point(x, gradient, hessian, delta):
    alpha = np.matmul(np.matmul(np.transpose(gradient(x)), hessian(x)), gradient(x))
    if alpha <= 0:
        tau = 1
    else:
        tau = min(np.linalg.norm(gradient(x), 2) / delta * np.matmul(np.matmul(np.transpose(gradient(x)), hessian(x)), gradient(x)), 1)

    return np.multiply(-tau * delta / np.linalg.norm(gradient(x), 2), gradient(x))


def trust_region(objective, gradient, hessian, iterationNumber, delta, callback, epsilon=0.01,  dimension=2):
    assert iterationNumber > 0, dimension > 0
    # initial point
    x = np.ones(dimension) * -2
    for i in range(iterationNumber):
        # find p
        p = cauchy_point(x, gradient, hessian, delta)
        # evaluating rho
        model = model_function(x, objective, gradient, hessian, p)
        print(model, p, x)
        rho = model / objective(x)
        if rho < 1/4:
            delta = 1/4 * np.linalg.norm(p)
        if rho > 3/4 and np.linalg.norm(p) == delta:
            delta = min(2*delta, 1.7)  # attention

        else:
            pass

        if rho > 0.2:
            x = x + p
        else:
            pass

        callback(objective(x), x, i)

    return x