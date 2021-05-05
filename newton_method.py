import numpy as np
import numpy.linalg as linalg
import line_search as ls


def newton_method(objective, gradient, hessain, iterationNumber, callback, epsilon=0.01, dimension=2):
    assert iterationNumber > 0, dimension > 0
    x = np.zeros(dimension)
    for i in range(iterationNumber):
        x = x - np.dot(linalg.inv(hessain(x)), gradient(x)) * ls.find_step_length(objective, gradient, x, 1, -gradient(x), 0.9)
        callback(objective(x), x, i)
        if np.linalg.norm(gradient(x)) < epsilon:
            break
    return x