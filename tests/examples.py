import numpy as np


def _f(Q, x, get_hessian=False):
    """ Returns xT Q x"""
    res = np.matmul(x.T, Q)
    res = np.matmul(res, x)

    # For a matrix [a,b;c,d]
    # Derivative is:  [2a, b+c ; b+c 2d]
    derv = np.matmul(x.T, 2*Q)

    # Compute Hessian
    if get_hessian:
        hessian = Q + Q.T
        return res, derv, hessian

    return res, derv, None


def f1(x, get_hessian=False):
    """
    :param x: vector x
    :return: The scalar function value evaluated at x
    :return: vector valued gradient at x
    """
    Q = np.identity(2)
    return _f(Q, x, get_hessian=get_hessian)


def f2(x,get_hessian=False):
    Q = np.identity(2)
    Q[0, 0] = 5
    return _f(Q, x, get_hessian=get_hessian)


def f3(x, get_hessian=False):
    a00 = 0.5 * np.sqrt(3)
    a01 = -0.5
    a10 = 0.5
    a11 = 0.5 * np.sqrt(3)

    A = np.array([[a00, a01], [a10, a11]])

    B = np.identity(2)
    B[0, 0] = 5

    Q = np.matmul(A.T, B)
    Q = np.matmul(Q, A)
    return _f(Q, x, get_hessian=get_hessian)

def rosenbrock(x, get_hessian=False):
    # https://www.wolframalpha.com/input/?i=100*%28y-x%5E2%29%5E2%2B%281-x%29%5E2+where+x%3D-1.4%2C+y%3D1.1
    x1, x2 = x
    a=1
    b = 100

    val = b * ((x2 - x1 ** 2) ** 2) + (1- x1) ** 2

    # 1st derivative:
    grad1 = 2 * (x1 - a) - 4 * b * x1 * (x2 - (x1 ** 2))
    grad2 = 2*b*(x2-x1**2)

    # Hessian:

    hess00 = 1200 * (x1**2)
    hess01 = hess10 = -400 * x1
    hess11 = 200
    if get_hessian:
        return val, np.array([grad1, grad2]), np.array( [  [hess00, hess01] ,[hess10, hess11]  ])

    return val, np.array([grad1,grad2]), None

def my_linear_func(x, get_hessian=False):
    a = np.array([-2,2])
    if get_hessian:
        return np.matmul(a.T, x), a, np.zeros(shape=(2, 2)), np.zeros((2,2))
    return np.matmul(a.T, x), a, None