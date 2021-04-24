import unittest
import logging
from src.utils import plot_path_contour

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    filename='test_report.txt',
    filemode='w'
)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.FileHandler())
# logger.addHandler(logging.StreamHandler(sys.stdout))
import numpy as np

from tests.examples import f1, f2, f3, rosenbrock, my_linear_func
from src.unconstrained_min import gradient_descent
from src.utils import FUNC2STR


class TestGradientDecsent(unittest.TestCase):

    def _test_func_impl(self):
        # f1 = x^2 + y^2
        # f2 = 5x^2 + y^2

        x = np.array([2, 2])
        a, b = f1(x)
        assert a == np.sum(np.power(x, 2))
        assert np.allclose(2 * x, b)

        a, b = f2(x)
        y = np.power(x, 2)
        assert a == 5 * y[0] + y[1]

    def _test_some_func(self, func):
        x_0 = np.array([1, 1])
        step_size = 0.1
        max_iteration = 100
        step_tol = 1e-8
        obj_tolerance = 1e-12

        logger.info(f'Testing function {FUNC2STR[func]} with parameters:\n'
                    f'initial point {x_0}\n'
                    f'step size {step_size}\n'
                    f'max number of iterations {max_iteration}\n'
                    f'step size tolerance {step_size}\n'
                    f'objective function tolerance {obj_tolerance}')

        is_sucess, last_point, X1, X2 = gradient_descent(f=func, x0=x_0, step_size=step_size, max_iter=max_iteration,
                                                         obj_tol=obj_tolerance, param_tol=step_tol)
        plot_path_contour(func, FUNC2STR[func], X1, X2)
        d = 'Success' if is_sucess else  'Failure'
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info(f"Results for function {FUNC2STR[func]}: {d}, last point is {last_point}")
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        assert is_sucess

    @unittest.expectedFailure
    def test_lin_min(self):
        x_0 = np.array([1, 1])
        step_size = 0.01
        max_iteration = 100
        step_tol = 1e-8
        obj_tolerance = 1e-7

        logger.info(f'Testing the linear function with parameters:\n'
                    f'initial point {x_0}\n'
                    f'step size {step_size}\n'
                    f'max number of iterations {max_iteration}\n'
                    f'step size tolerance {step_size}\n'
                    f'objective function tolerance {obj_tolerance}')
        logger.info('This test is expected to fail as gradient descent will not find the minimum for a linear function')
        logger.info(
            'The only way this function can succeed, is if the tolerance parameters are higher than the step size\n'
            'This is misleading this the minimum point is not actually found')

        is_sucess, last_point, X1, X2 = gradient_descent(f=my_linear_func, x0=x_0, step_size=step_size,
                                                         max_iter=max_iteration,
                                                         obj_tol=obj_tolerance, param_tol=step_tol)
        plot_path_contour(my_linear_func, FUNC2STR[my_linear_func], X1, X2)
        d = 'Success' if is_sucess else  'Failure'
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info(f"Results for function {FUNC2STR[my_linear_func]}: {d}, last point is {last_point}")
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        assert is_sucess

    def test_quad_min(self):
        logger.info('start test_quad_min')

        for f in [f1, f2, f3]:
            self._test_some_func(f)

    def test_rosenbrock_min(self):
        x_0 = np.array([2, 2])
        step_size = 0.001
        max_iteration = 10000
        step_tol = 1e-8
        obj_tolerance = 1e-7
        logger.info(f'Testing function {FUNC2STR[rosenbrock]} with parameters:\n'
                    f'initial point {x_0}\n'
                    f'step size {step_size}\n'
                    f'max number of iterations {max_iteration}\n'
                    f'step size tolerance {step_size}\n'
                    f'objective function tolerance {obj_tolerance}')

        is_sucess, last_point, X1, X2 = gradient_descent(f=rosenbrock, x0=x_0, step_size=step_size,
                                                         max_iter=max_iteration,
                                                         obj_tol=obj_tolerance, param_tol=step_tol)

        plot_path_contour(rosenbrock, FUNC2STR[rosenbrock], X1, X2)

        d = 'Success' if is_sucess else  'Failure'
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info(f"Results for function {FUNC2STR[rosenbrock]}: {d}, last point is {last_point}")
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #assert np.allclose(last_point, np.array([1,1]), atol=obj_tolerance)


if __name__ == '__main__':
    unittest.main()
