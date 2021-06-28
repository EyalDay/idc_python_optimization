import logging
from functools import partial

import numpy as np

from src.utils import report_step

logger = logging.getLogger(__name__)


def grad_dir(current_grad,
             **ignore):  # **ignore is so the dir functions will have a uniform interface. Helps simplify the main loop.
    return -1 * current_grad


def newton_dir(current_hess, current_grad):
    return np.linalg.solve(current_hess, -1 * current_grad)


def bfgs_dir(current_hess, current_grad):
    return np.linalg.solve(current_hess, -1 * current_grad)


def update_Bk(Bk, current_point, next_point, current_grad, next_grad):
    sk = (next_point - current_point).reshape(-1, 1)
    yk = (next_grad - current_grad).reshape(-1, 1)
    return Bk - Bk @ sk @ sk.T @ Bk.T / (sk.T @ Bk @ sk) + (yk @ yk.T) / (yk.T @ sk)


def wolfe_condition(f, current_point, pk, init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.2):
    """
    Returns the step size according to the first Wolfe condition
    """

    def _is_cond_satisfied(current_point, step_len, pk, slope_ratio):
        """
        Checks if the 1st Wolfe condition is satisfied.
        """
        fk_1, _, _ = f(current_point + step_len * pk)
        fk, grad_fk, _ = f(current_point)
        return fk_1 <= fk + slope_ratio * step_len * grad_fk.T @ pk

    step_len = init_step_len
    while np.isnan(f(current_point + step_len * pk)[0]) or not _is_cond_satisfied(current_point, step_len, pk,
                                                                                  slope_ratio):
        step_len = back_track_factor * step_len
    return step_len


DIR_METHOD_DICT = {
    'gd': grad_dir,
    'nt': newton_dir,
    'bfgs': bfgs_dir

}


def line_search(f, x0, obj_tol, param_tol, max_iter, dir_selection_method='gd', init_step_len=1.0, slope_ratio=1e-4,
                back_track_factor=0.2):
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
    get_hessian = (dir_selection_method != 'gd')  # Indicates whether the hessian should be computed
    f = partial(f, get_hessian=get_hessian)
    current_val, current_grad, current_hessian = f(current_point)
    Bk = current_hessian
    dir_func = DIR_METHOD_DICT.get(dir_selection_method, None)

    if not dir_func:
        err_str = f"{dir_selection_method} should be one of 'gd', 'nt', 'bfgs'"
        logger.error(err_str)
        return err_str

    X1 = list()
    X2 = list()
    Y = list()
    while True:
        X1.append(current_point[0])
        X2.append(current_point[1])
        Y.append(current_val)
        pk = dir_func(current_grad=current_grad, current_hess=current_hessian)
        alpha = wolfe_condition(f=f, current_point=current_point, pk=pk, init_step_len=init_step_len,
                                slope_ratio=slope_ratio, back_track_factor=back_track_factor)

        next_point = current_point + alpha * pk
        next_val, next_grad, next_hess = f(next_point)

        report_step(i=iterations, x_i=current_point, x_i_1=next_point, fxi=current_val, fxi1=next_val, f=f,
                    method=dir_selection_method)

        abs_diff = np.abs(next_val - current_val)
        # print('abs_diff ', abs_diff, ' obj_tol ', obj_tol)
        if abs_diff <= obj_tol:
            logger.info(
                f'SUCCESS! Absolute difference in objective function values between two consecutive iterations is {abs_diff}\n'
                f'which is less than the objactive tolerance {obj_tol}'
                f'Took {iterations} iterations')
            return True, next_point, X1, X2, Y

        abs_dist = np.linalg.norm(current_point - next_point)
        print('abs_dist ', abs_dist, ' param_tol ', param_tol)
        if abs_dist <= param_tol:
            logger.info(f'SUCCESS! Distance between two consecutive iterations iteration locations is {abs_dist}\n'
                        f'which is less than the objactive tolerance {param_tol}\n'
                        f'Took {iterations} iterations')
            return True, next_point, X1, X2, Y

        if iterations == max_iter:
            logger.info(f"FAIULRE! Did not converge in {iterations} iterations")
            return False, next_point, X1, X2, Y

        iterations += 1

        if dir_selection_method == "bfgs":
            Bk = update_Bk(Bk=Bk, current_point=current_point, next_point=next_point, current_grad=current_grad,
                           next_grad=next_grad)

        current_point, current_val = next_point, next_val
