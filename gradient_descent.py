import numpy as np
import line_search as ls


def gradient_descent(objective, gradient, iterationNumber, callback, epsilon=0.1, dimension=2):
    assert iterationNumber > 0, dimension > 0
    x = np.zeros(dimension)

    for i in range(iterationNumber):
        x = x - gradient(x) * ls.find_step_length(objective, gradient, x, 1, -gradient(x), 0.9)
        callback(objective(x), i)
        if np.linalg.norm(gradient(x)) < epsilon:
            break
    return x
