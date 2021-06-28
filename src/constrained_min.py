from src.utils import report_step
from functools import partial
from src.unconstrained_min import newton_dir, wolfe_condition
import numpy as np
import logging

logger = logging.getLogger(__name__)


def interior_pt(f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, init_step_len=1.0, slope_ratio=1e-4,
                back_track_factor=0.2):
    m, t, eps = len(ineq_constraints), 1, 1e-6
    path = list()
    current_point = x0
    while m / t > eps:  # Lecture 9, slides 12 - 15
        lambda_x = 100 * eps
        iterations = 0
        while 0.5 * lambda_x > eps:
            constrained_f = partial(barrier_func, t=t, f=f, ineq_constraints=ineq_constraints)
            y, grad, hess = constrained_f(current_point=current_point)
            path.append((current_point, y, 'inner'))
            if eq_constraints_mat is not None:
                pk = constrained_newton_dir(A=eq_constraints_mat, B=eq_constraints_rhs, grad=grad, hess=hess)
            else:
                pk = newton_dir(current_grad=grad, current_hess=hess)

            lambda_x = (pk.T @ hess @ pk).item()
            alpha = wolfe_condition(f=constrained_f, current_point=current_point, pk=pk, init_step_len=init_step_len,
                                    slope_ratio=slope_ratio, back_track_factor=back_track_factor)

            current_point = current_point + alpha * pk
            iterations += 1
            if iterations > 50 or alpha < eps:
                break

        path.append((current_point, y, 'outer'))
        t *= 10  # increase by a factor of 10 in each outer iteration

    return path


def barrier_constraint(constraint, current_point):
    '''
    Given an ineq constraint and a point, returns the value, grad and hess of the log barrier at that point
    See book https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
    Page 567 (pdf 578)
    '''
    fx, grad, hess = constraint(x=current_point)
    ret_fx = np.log(-fx) if fx < 0 else np.infty
    ret_grad = -grad / fx
    ret_hess = grad @ grad.T / (fx * fx) - (hess * hess) / fx  # bug was missing (-) at the denominator
    return ret_fx, ret_grad, ret_hess


def barrier_func(current_point, t, f, ineq_constraints):
    y, grad, hess = f(x=current_point)
    y, grad, hess = t * y, t * grad, t * hess
    for constraint in ineq_constraints:
        y_, grad_, hess_ = barrier_constraint(constraint, current_point)
        y, grad, hess = y + y_, grad + grad_, hess + hess_
    return y, grad, hess


def constrained_newton_dir(A, B, grad, hess):
    # Lecture 7, slide 19

    m, n = A.shape

    L = np.vstack((np.hstack((hess, A.T)), np.hstack((A, np.zeros((m, m))))))
    # L.shape = 4,4

    R = np.vstack((-grad, np.zeros((m, 1))))
    v = np.linalg.solve(L, R)

    return v[:n]
