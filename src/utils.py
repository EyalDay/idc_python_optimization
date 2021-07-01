import logging

from matplotlib import cm

from tests.examples import *

logger = logging.getLogger(__name__)
import numpy as np
import matplotlib.pyplot as plt

i = 1
FUNC2STR = {
    f1: '1.Quadratic function xT*[1,0;0,1]*x',
    f2: '2.Quadratic function xT*[5,0;0,1]*x',
    f3: '3.Quadratic function xT*Q*x',
    rosenbrock: '4.￿Rosenbrock',
    my_linear_func: '3.d, ￿Linear function f(x)=[-2,2]T*x'
}
def report_step(i, x_i, x_i_1=None, fxi=None, fxi1=None, f=None, method=None):
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
    if x_i_1 is not None:
        step_len = np.linalg.norm(x_i_1 - x_i)
    else:
        step_len = np.nan
    if fxi1 is not None and fxi is not None:
        dfx = fxi1 - fxi
    else:
        step_len = np.nan

    logger.info(
        f'method {method} iteration {i}: Locations{x_i} -> {x_i_1} Step length = {step_len} Function values: {fxi} -> {fxi1} delta: {dfx}')


def _get_Z(func):
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    if func == rosenbrock:
        x_min, x_max = -2, 5
        y_min, y_max = -2, 5
    if func == my_linear_func:
        x_min, x_max = 0.9, 3.1
        y_min, y_max = -1.1, 1.1

    delta = 0.01
    x1 = np.arange(x_min, x_max, delta)
    x2 = np.arange(y_min, y_max, delta)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.empty_like(X1)
    w, h = Z.shape
    for i in range(w):
        for j in range(h):
            val, _, _ = func(np.array([X1[i, j], X2[i, j]]), get_hessian=False)
            Z[i, j] = val

    return X1, X2, Z


def plot_path_contour(func, title, P1=None, P2=None, method=None):
    """
    path taken by the algorithm should be plotted, overlaid on a plot of function contours
    :return:
    """
    fig, ax = plt.subplots()

    colors = cm.jet(np.linspace(0, 1, len(P1)))
    ax.scatter(P1, P2, s=2, c=colors)
    ax.scatter(P1[0], P2[0], label='initial point')
    ax.scatter(P1[-1], P2[-1], label='final point')

    # Plot contours
    X1, X2, Z = _get_Z(func)
    min_x, max_x = np.min(X1) - 0.2, np.max(X1) + 0.2
    min_y, max_y = np.min(X2) - 0.2, np.max(X2) + 0.2


    xx = np.linspace(min_x, max_x, 50)
    yy = np.linspace(min_y, max_y, 50)
    XX,YY = np.meshgrid(xx,yy)

    PTS = np.vstack((np.ravel(XX), np.ravel(YY))).T
    Z = np.array([func(pt)[0] for pt in PTS]).reshape(XX.shape)
    #lables = np.arange(np.min(Z), np.max(Z), 10)
    CS = ax.contour(XX, YY, Z, levels=10)
    ax.clabel(CS, fontsize=9, inline=True)

    # Title, axis limits, etc.
    ax.set_title(f'method {method} Function {title}')

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.set_xlabel('$x_1$')
    ax.set_xlabel('$x_2$')

    # Save plt
    global i
    plt.legend()
    plt.savefig(fname=f'{method}_{i}' + '.png')
    # plt.show()


def plot_obj_value(method, Y, func=None):
    global i
    plt.figure()
    plt.plot(Y)
    plt.title(f'Method {method} function {func}')
    plt.xlabel("Iteration number")
    plt.ylabel("Objective value")
    if func == 'rosen':
        plt.yscale('log')
        plt.xticks(list(range(len(Y))))

    plt.savefig(fname=f'{method}{func or ""}_{i}_converge' + '.png')
    i += 1

