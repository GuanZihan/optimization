import numpy as np
import line_search as ls


def model_function(x, objective, gradient, hessian,  p):
    a = np.multiply(np.multiply(np.transpose(p), hessian(x)), p)
    return objective(x) + np.matmul(np.transpose(gradient(x)), p) + 1/2 * np.matmul(np.matmul(np.transpose(p), hessian(x)), p)


def trust_region(objective, gradient, hessian, iterationNumber, callback, epsilon=0.01, dimension=2):
    assert iterationNumber > 0, dimension > 0
    delta = 1
    x = np.ones(dimension) * -2
    # find p
    p = np.array([[-2],[-2]]) # attention
    print(np.shape(np.transpose(p)))
    # evaluating rho
    model = model_function(x, objective, gradient, hessian, p)
    rho = model / objective(x)
    if rho < 1/4:
        delta = 1/4 * np.linalg.norm(p)
    if rho > 3/4 and np.linalg.norm(p) == delta:
        delta = min(2*delta, 1)  # attention
    else:
        pass

    if rho > 1/4:
        x = x + p
    else:
        pass

    return x