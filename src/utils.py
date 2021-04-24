import logging

from matplotlib import cm

from tests.examples import *

logger = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt

i = 1
FUNC2STR = {
    f1: '3.b.i, Quadratic function xT * [1,0;0,1] * x',
    f2: '3.b.ii, Quadratic function xT * [5,0;0,1] * x',
    f3: '3.b.iii, Quadratic function f(x) = xT * Q * x',
    rosenbrock: '3.c, ￿Rosenbrock',
    my_linear_func: '3.d, ￿Linear function f(x)=[-2,2]T*x'
}

FUNC2AX = {
    f1: ([-1.5, 1.5], [-1.5, 1.5]),
    f2: ([-1.1, 1.1], [-1.2, 1.2]),
    f3: ([-1.1, 1.1], [-1.2, 1.2]),
    rosenbrock: ([-0.5, 3], [-0.5, 3]),
    my_linear_func: ([0.9, 3], [-1.1, 1.1])
}


def report_step(i, x_i, x_i_1, fxi, fxi1, f):
    """
    :param i: Iteration step
    :param x_i: Location at step i
    :param x_i_1: Location at step i+1
    :param f: Function
    :return:
    """

    # iteration number
    # current location x0
    # current objactive value f(x0)
    # step length |x1 - x0|
    # change in objective function value
    step_len = np.linalg.norm(x_i_1 - x_i)
    dfx = fxi1 - fxi
    logger.info(
        f'iteration {i}: Locations{x_i} -> {x_i_1} Step length = {step_len} Function values: {fxi} -> {fxi1} delta: {dfx}')


def _get_Z(func):
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    if func ==rosenbrock:
        x_min, x_max = -2, 5
        y_min, y_max = -2, 5
    if func ==my_linear_func:
        x_min, x_max = 0.9, 3.1
        y_min, y_max = -1.1,1.1


    delta = 0.01
    x1 = np.arange(x_min, x_max, delta)
    x2 = np.arange(y_min, y_max, delta)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.empty_like(X1)
    w, h = Z.shape
    for i in range(w):
        for j in range(h):
            val, _ = func(np.array([X1[i, j], X2[i, j]]))
            Z[i, j] = val

    return X1, X2, Z


def plot_path_contour(func, title, P1=None, P2=None):
    """
    path taken by the algorithm should be plotted, overlaid on a plot of function contours
    :return:
    """
    X1, X2, Z = _get_Z(func)

    colors = cm.jet(np.linspace(0, 1, len(P1)))

    fig, ax = plt.subplots()

    ax.scatter(P1, P2, s=2, c=colors)

    if func == rosenbrock:
        levels = np.sort(np.unique(np.concatenate([np.arange(0, 100, 10), np.arange(100, 800, 30)])))
    elif func == my_linear_func:
        levels = np.arange(-10, 10, 0.5)
    else:
        levels = np.sort(np.unique(np.concatenate([np.arange(0, 2, 0.25), np.arange(0, 1, 0.1)])))

    CS = ax.contour(X1, X2, Z, levels)

    ax.clabel(CS, fontsize=9, inline=True)
    ax.set_title(f'Function {title}')
    x_lim, y_lim = FUNC2AX[func]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    global i
    plt.savefig(fname=str(i) + '.png')
    i += 1
    plt.show()

