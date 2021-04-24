import numpy as np
from functools import partial


def _f(Q, x):
    """ Returns xT Q x"""
    res = np.matmul(x.T, Q)
    res = np.matmul(res, x)

    # For a matrix [a,b;c,d]
    # Derivative is:  [2a, b+c ; b+c 2d]
    derv = np.matmul(x.T, 2*Q)
    return res, derv


def f1(x):
    """
    :param x: vector x
    :return: The scalar function value evaluated at x
    :return: vector valued gradient at x
    """
    Q = np.eye(2)
    return _f(Q, x)


def f2(x):
    Q = np.eye(2)
    Q[0, 0] = 5
    return _f(Q, x)


def f3(x):
    a00 = 0.5 * np.sqrt(3)
    a01 = -0.5
    a10 = 0.5
    a11 = 0.5 * np.sqrt(3)

    A = np.array([[a00, a01], [a10, a11]])

    B = np.eye(2)
    B[0, 0] = 5

    Q = np.matmul(A.T, B)
    Q = np.matmul(Q, A)
    return _f(Q, x)

def rosenbrock(x):
    # https://www.wolframalpha.com/input/?i=100*%28y-x%5E2%29%5E2%2B%281-x%29%5E2+where+x%3D-1.4%2C+y%3D1.1
    x1, x2 = x
    b = 100

    val = b * ((x2 - x1 ** 2) ** 2) + (1- x1) ** 2

    grad1 = 2*(200*np.power(x1,3) - 200 * x1 * x2 -1)
    grad2 = 200*(x2-x1**2)

    return val, np.array([grad1,grad2])

def my_linear_func(x):
    a = np.array([-2,2])
    return np.matmul(a.T, x), a