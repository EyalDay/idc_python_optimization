from tests.examples import *

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

FUNC2STR  = {
    f1: '3.b.i, quadratic function xT [1,0;0,1] x',
    f2: '3.b.ii, quadratic function xT [5,0;0,1] x',
    f3: '3.b.iii, quadratic function f(x) = xT Q x',
    rosenbrock: '3.c, rosenbrock',
    my_linear_func: '3.d, linear function f(x)=[-2,2]Tx'
}

def report_step(i, x_i, x_i_1, f ):
    """

    :param i: Iteration step
    :param x_i: Location at step i
    :param x_i_1: Location at step i-1
    :param f: Function
    :return:
    """
    print(f'function {FUNC2STR[f]} iteration {i}: {x_i} -> {x_i_1}')

def _get_Z(func):
    x_min,x_max = -2,2
    y_min,y_max = -2,2
    delta = 0.01
    x1 = np.arange(x_min,x_max, delta)
    x2 = np.arange(y_min,y_max, delta)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.empty_like(X1)
    w,h = Z.shape
    for i in range(w):
        for j in range(h):
            val, _ = func(np.array([X1[i,j], X2[i,j]]))
            Z[i,j] = val

    return X1,X2,Z

def plot_path_contour(func, func_points = None):
    """
    path taken by the algorithm should be plotted, overlaid on a plot of function contours
    :return:
    """
    X1, X2, Z = _get_Z(func)

    #x,y = func_points
    #plt.scatter(x, y)

    fig, ax = plt.subplots()

    levels = np.arange(0, 2, 0.25)

    CS = ax.contour(X1, X2, Z, levels)
    ax.clabel(CS, fontsize=9, inline=True)
    ax.set_title('Single color - negative contours dashed')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    plt.show()


'''
import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-3.0, 3.0, 100)
ylist = np.linspace(-3.0, 3.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot')
#ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
plt.show()

'''