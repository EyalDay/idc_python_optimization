import unittest
import logging
import sys
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.FileHandler('test_report.txt'))
logger.addHandler(logging.StreamHandler(sys.stdout))
import numpy as np

from tests.examples import f1,f2,f3,rosenbrock,my_linear_func
from src.unconstrained_min import gradient_descent
from src.utils import FUNC2STR

class TestGradientDecsent(unittest.TestCase):

    def _test_func_impl(self):
        # f1 = x^2 + y^2
        # f2 = 5x^2 + y^2

        x = np.array([2,2])
        a,b = f1(x)
        assert a == np.sum(np.power(x,2))
        assert np.allclose(2*x, b)

        a, b = f2(x)
        y = np.power(x,2)
        assert a == 5*y[0] + y[1]

    def _test_some_func(self, func):
        x_0 = np.array([1,1])
        step_size = 0.1
        max_iteration = 100
        step_tol = 1e-8
        obj_tolerance = 1e-12

        is_sucess, last_point = gradient_descent(f=func,x0=x_0,step_size=step_size,max_iter=max_iteration,obj_tol=obj_tolerance, param_tol=step_tol)
        print(f"{FUNC2STR[func]}: {is_sucess} {last_point}")

    def test_quad_min(self):
        logger.info('start test_quad_min')

        for f in [f1,f2,f3]:
            self._test_some_func(f)


    def _test_rosenbrock_min(self):
        x_0 = np.array([2,2])
        step_size = 0.001
        max_iteration = 10000
        step_tol = 1e-8
        obj_tolerance = 1e-7
        is_sucess, last_point = gradient_descent(f=rosenbrock, x0=x_0, step_size=step_size, max_iter=max_iteration,
                                                 obj_tol=obj_tolerance, param_tol=step_tol)
        print(f"{FUNC2STR[rosenbrock]}: {is_sucess} {last_point}")
        #assert np.allclose(last_point, np.array([1,1]), atol=obj_tolerance)

    def _test_lin_min(self):
        x_0 = np.array([2,2])
        step_size = 0.001
        max_iteration = 100
        step_tol = 1e-8
        obj_tolerance = 1e-7
        is_sucess, last_point = gradient_descent(f=my_linear_func, x0=x_0, step_size=step_size, max_iter=max_iteration,
                                                 obj_tol=obj_tolerance, param_tol=step_tol)
        print(f"{FUNC2STR[my_linear_func]}: {is_sucess} {last_point}")

if __name__ == '__main__':
    unittest.main()
