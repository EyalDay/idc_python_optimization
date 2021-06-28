import logging
import unittest

import numpy as np

from src.unconstrained_min import line_search
from src.utils import plot_path_contour, plot_obj_value, FUNC2STR
from tests.examples import f1, f2, f3, rosenbrock, my_linear_func

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

class TestUnconstrainedMin(unittest.TestCase):

    def _test_some_func(self, func, method):
        x_0 = np.array([1, 1])
        step_size = 0

        max_iteration = 100
        step_tol = 1e-8
        obj_tolerance = 1e-12
        init_step_len = 1.0
        slope_ratio = 1e-4
        back_track_factor = 0.2

        logger.info(f'Testing function {FUNC2STR[func]} with method {method} and parameters:\n'
                    f'initial point {x_0}\n'
                    f'max number of iterations {max_iteration}\n'
                    f'step size tolerance {step_size}\n'
                    f'objective function tolerance {obj_tolerance}\n'
                    f'init_step_len {init_step_len}\nslope_ratio{slope_ratio}\nback_track_factor {back_track_factor}'
                    )

        is_success, last_point, X1, X2, Y = line_search(f=func, x0=x_0, max_iter=max_iteration,
                                                         obj_tol=obj_tolerance, param_tol=step_tol, dir_selection_method='bfgs',
                                                        init_step_len=init_step_len, slope_ratio=slope_ratio, back_track_factor=back_track_factor)
        plot_path_contour(func, FUNC2STR[func], X1, X2, method)
        plot_obj_value(title=FUNC2STR[func], method=method, Y=Y)
        d = 'Success' if is_success else 'Failure'
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info(f"Results for function {FUNC2STR[func]}: {d}, last point is {last_point}")
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        assert is_success
        assert np.allclose(last_point, np.array([0,0]), atol=obj_tolerance)

    @unittest.expectedFailure
    def _test_lin_min(self):
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

        is_success, last_point, X1, X2, Y = line_search(f=my_linear_func, x0=x_0,
                                                         max_iter=max_iteration,
                                                         obj_tol=obj_tolerance, param_tol=step_tol, dir_selection_method='bfgs')
        plot_path_contour(my_linear_func, FUNC2STR[my_linear_func], X1, X2)
        d = 'Success' if is_success else 'Failure'
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        logger.info(f"Results for function {FUNC2STR[my_linear_func]}: {d}, last point is {last_point}")
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        assert is_success

    def test_quad_min(self):
        logger.info('start test_quad_min')
        for method in ['gd', 'nt', 'bfgs']:
            for f in [f1, f2, f3]:
                self._test_some_func(f, method)

    def test_rosenbrock_min(self):
        for method in ['gd', 'nt', 'bfgs']:
            x_0 = np.array([2, 2])
            max_iteration = 10000
            step_tol = 1e-8
            obj_tolerance = 1e-7
            logger.info(f'Testing function {FUNC2STR[rosenbrock]} with parameters:\n'
                        f'method {method}\n'
                        f'initial point {x_0}\n'
                        f'max number of iterations {max_iteration}\n'
                        f'step size tolerance {step_tol}\n'
                        f'objective function tolerance {obj_tolerance}')

            is_success, last_point, X1, X2, Y = line_search(f=rosenbrock, x0=x_0,
                                                             max_iter=max_iteration,
                                                             obj_tol=obj_tolerance, param_tol=step_tol, dir_selection_method='bfgs')

            plot_path_contour(rosenbrock, FUNC2STR[rosenbrock], X1, X2)
            plot_obj_value(title=FUNC2STR[rosenbrock], func = 'rosen', method=method, Y=Y)

            d = 'Success' if is_success else  'Failure'
            logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            logger.info(f"Results for function {FUNC2STR[rosenbrock]}: {d}, last point is {last_point}")
            logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            assert is_success


if __name__ == '__main__':
    unittest.main()
