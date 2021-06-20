from src.utils import report_step
import numpy as np
import logging

logger = logging.getLogger(__name__)

def grad_dir(current_point, step_size, grad):
    return current_point - step_size * grad

def bfgs_dir():
    pass

def newton_dir():
    pass


DIR_METHOD_DICT = {
    'gd': grad_dir,
    'nt': newton_dir,
    'bfgs': bfgs_dir

}

def line_search(f, x0, step_size, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len, slope_ratio, back_track_factor):
    """
    :param f: Function minimized
    :param x0: Starting point
    :param step_size: coefficient multiplying the gradient vector in the algorithm update rule
    :param obj_tol: numeric tolerance for successful termination in terms of small enough change in objective function values between two consecutive iterations
    :param param_tol: numeric tolerance for successful termination in terms of small enough distance between two consecutive iterations iteration locations
    :param max_iter: maximum allowed number of iterations
    :param dir_selection_method: Direction for line search ('gd' for gradient descent, 'nt' for Newton or 'bfgs' for BFGS)
    :param init_step_len, slope_ratio, back_track_factor: backtracking and Wolfe condition parameters

    :return:  Last location and a success/failure Boolean flag, according to the termination conditions.
    """

    current_point = x0
    iterations = 1
    current_val, grad = f(current_point)
    dir_func = DIR_METHOD_DICT.get(dir_selection_method,None)
    if not dir_func:
        err_str = f"{dir_selection_method} should be one of 'gd', 'nt', 'bfgs'"
        logger.error(err_str)
        return err_str
    X1 = list()
    X2 = list()
    while True:
        X1.append(current_point[0])
        X2.append(current_point[1])
        next_point = dir_func(...)
        next_val, grad = f(next_point)

        report_step(i=iterations, x_i=current_point, x_i_1=next_point, fxi =current_val, fxi1 = next_val, f=f)

        abs_diff = np.abs(next_val - current_val)
        if abs_diff <= obj_tol:
            logger.info(
                f'SUCCESS! Absolute difference in objective function values between two consecutive iterations is {abs_diff}\n'
                f'which is less than the objactive tolerance {obj_tol}'
                f'Took {iterations} iterations')
            return True, next_point, X1, X2

        abs_dist = np.linalg.norm(current_point - next_point)
        if abs_dist <= param_tol:
            logger.info(f'SUCCESS! Distance between two consecutive iterations iteration locations is {abs_dist}\n'
                        f'which is less than the objactive tolerance {param_tol}\n'
                        f'Took {iterations} iterations')
            return True, next_point, X1, X2

        if iterations == max_iter:
            logger.info(f"FAIULRE! Did not converge in {iterations} iterations")
            return False, next_point, X1, X2

        iterations += 1


        current_point, current_val = next_point, next_val
