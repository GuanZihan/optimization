import numpy as np
import numpy.linalg as linalg
import line_search as ls


def newton_method(objective, gradient, hessian, iterationNumber, callback, epsilon=0.001, dimension=2):
    assert iterationNumber > 0, dimension > 0
    x = np.ones(dimension) * -2
    for i in range(iterationNumber):
        direction = -linalg.solve(hessian(x), gradient(x))
        x_prev = x
        x = x + direction * ls.find_step_length(objective, gradient, x, 1, direction, 0.9)
        callback(objective(x), x, i)
        if np.linalg.norm(gradient(x)) < epsilon:
            break
    return x