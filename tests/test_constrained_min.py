import unittest
import numpy as np
from functools import partial
from src import constrained_min
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class TestConstrainedMin(unittest.TestCase):

    @staticmethod
    def linear_func(a, x, b=0, **kwargs):
        """
        linear func ax +b . Returns value, grad and hess
        :param a: nx1 column vector
        :param x: 1xn row vector
        """
        ret_v, ret_grad, ret_hess = (a.T @ x + b).item(), a, np.zeros(shape=(a.shape[0], a.shape[0]))

        assert not isinstance(ret_v, np.ndarray)
        assert ret_grad.shape == x.shape
        assert ret_hess.shape == (x.shape[0], x.shape[0])

        return ret_v, ret_grad, ret_hess

    @staticmethod
    def quad_func(x, a, b, c, get_hessian=False):
        """
        Quadratic func ax^2 + bx + c. Returns value, grad and hess

        """
        # shapes:
        # x column vector 3,1
        # a mat 3,3
        # b column vector 3,1
        # c const
        ret_v, ret_grad = (x.T @ a @ x + b.T @ x + c).item(), (a + a.T) @ x + b
        ret_hess = a + a.T if get_hessian else None
        assert not isinstance(ret_v, np.ndarray)
        assert ret_grad.shape == (3, 1)
        assert ret_hess is None or ret_hess.shape == (3, 3)

        return ret_v, ret_grad, ret_hess

    def test_qp(self):
        # https://www.wolframalpha.com/input/?i=minimize+x%5E2%2By%5E2%2B%28z%2B1%29%5E2+subject+to+x%2By%2Bz%3D1+and+x%3E%3D0+and+y%3E%3D0+and+z%3E%3D0+
        a = np.array([-1, 0, 0]).reshape(3, 1)
        # Inequality constraints: -xi < 0 for i=0,1,2 -> xi >= 0 as asked
        ineq_constraints = [partial(TestConstrainedMin.linear_func, a=np.roll(a, i)) for i in range(3)]
        # Equality constraints: x1+x2+x3 = 1
        # Ax=b -> A.shape = m,3 , x.shape = 3,1, b.shape = m,1
        # In this case we have m=1 constraints
        eq_constraints_mat = np.array([1, 1, 1]).reshape(1, 3)
        eq_constraints_rhs = np.array(1)
        # Initial interior point 0.1,.02,0.7
        x0 = np.array([0.1, 0.2, 0.7]).reshape(-1, 1)

        # Function: x1^2 + x2^2 +(x3+1)^2 = x1^2 + x2^2 +x3^2 + 2x3 + 1
        f = partial(TestConstrainedMin.quad_func, a=np.identity(3), b=np.array([0, 0, 2]).reshape((-1, 1)), c=1,
                    get_hessian=True)
        path = constrained_min.interior_pt(f, ineq_constraints,
                                           eq_constraints_mat, eq_constraints_rhs, x0)

        pts = [p[0] for p in path]
        outer_pts = [p[0] for p in path if p[2] == 'outer']
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', xlim=[0, 1], ylim=[0, 1], zlim=[0, 1])

        # Plot feasible region. Reference https://matplotlib.org/2.0.2/mpl_toolkits/mplot3d/tutorial.html#surface-plots
        X = np.linspace(0, 1, 50)
        Y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(X, Y)
        Z = 1 - X - Y
        ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha=0.3)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Plot the path:
        ax.plot(xs=[pt[0].item() for pt in pts], ys=[pt[1].item() for pt in pts], zs=[pt[2].item() for pt in pts],
                label='Path taken')
        # Mark the starting point
        start = pts[0]
        ax.scatter(xs=[start[0].item()], ys=[start[1].item()], zs=[start[2].item()],
                   label=f'start ({start[0].item():.3f}, {start[1].item():.3f}, {start[2].item():.3f})', color='green')
        # Mark the finishing point
        end = pts[-1]
        ax.scatter(xs=[end[0].item()], ys=[end[1].item()], zs=[end[2].item()],
                   label=f'end ({end[0].item():.3f}, {end[1].item():.3f}, {end[2].item():.3f})', color='red')

        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-0.1, 1.01)
        ax.set_ylim(-0.1, 1.01)
        ax.set_zlim(-0.1, 1.01)

        plt.legend()
        plt.show()

    def test_lp(self):
        ineq_constraints = [
            partial(TestConstrainedMin.linear_func, a=np.array([-1, -1]).reshape(-1, 1), b=1),
            partial(TestConstrainedMin.linear_func, a=np.array([0, 1]).reshape(-1, 1), b=-1),
            partial(TestConstrainedMin.linear_func, a=np.array([1, 0]).reshape(-1, 1), b=-2),
            partial(TestConstrainedMin.linear_func, a=np.array([0, -1]).reshape(-1, 1), b=0)
        ]
        eq_constraints_mat = None
        eq_constraints_rhs = None
        x0 = np.array([0.5, 0.75]).reshape(-1, 1)
        f = partial(TestConstrainedMin.linear_func, a=np.array([1, 1]).reshape(-1, 1))

        path = constrained_min.interior_pt(f=f, ineq_constraints=ineq_constraints,
                                           eq_constraints_mat=eq_constraints_mat, eq_constraints_rhs=eq_constraints_rhs,
                                           x0=x0)


        x_lims = (0-0.1,2+0.1)
        y_lims = (0-0.1,1+0.1)

        pts = [pt[0] for pt in path]
        print(path)
        X = [pt[0].item() for pt in pts]
        Y = [pt[1].item() for pt in pts]
        plt.figure()
        # plot the feasible region
        x = np.linspace(*x_lims, 900)
        y = np.linspace(*y_lims, 900)
        x, y = np.meshgrid(x, y)
        plt.imshow(((y <= 1) &
                    (y >= 0) &
                    (x <= 2) &
                    (y >= -x + 1)),
                   extent=(*x_lims, *y_lims), origin="lower", cmap="Greys", alpha=0.3, label='feasible region')

        # Plot the path:
        plt.plot(X, Y, label='Path taken', color='blue')
        plt.scatter(x=X, y=Y, color='blue')

        # Mark the starting point
        start = pts[0]
        print('start', start)
        plt.scatter(x=[start[0].item()], y=[start[1].item()],
                   label=f'start ({start[0].item():.3f}, {start[1].item():.3f}', color='green')
        # Mark the finishing point
        end = pts[-1]
        print('end', end)

        plt.scatter(x=[end[0].item()], y=[end[1].item()],
                   label=f'end ({end[0].item():.3f}, {end[1].item():.3f}', color='red')



        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(*x_lims)
        plt.ylim(*y_lims)
        plt.grid(True, which='both')
        plt.legend()

        plt.show()
