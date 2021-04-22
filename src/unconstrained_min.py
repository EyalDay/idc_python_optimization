from src.utils import report_step
import numpy as np

def gradient_descent(f, x0, step_size, obj_tol, param_tol,max_iter):
    """
    :param f: Function minimized
    :param x0: Starting point
    :param step_size: coefficient multiplying the gradient vector in the algorithm update rule
    :param obj_tol: numeric tolerance for successful termination in terms of small enough change in objective function values between two consecutive iterations
    :param param_tol: numeric tolerance for successful termination in terms of small enough distance between two consecutive iterations iteration locations
    :param max_iter: maximum allowed number of iterations
    :return:  Last location and a success/failure Boolean flag, according to the termination conditions.
    """

    current_point = x0
    iterations = 1
    current_val, grad = f(current_point)
    while True:
        next_point = current_point - step_size * grad
        next_val, grad = f(next_point)

        report_step(i=iterations,x_i=current_point, x_i_1=next_point, f= f)

        abs_diff = np.abs(next_val - current_val)
        if abs_diff <= obj_tol:
           print(f'SUCCESS! Absolute difference in objective function values between two consecutive iterations is {abs_diff}\n' 
                 f'which is less than the objactive tolerance {obj_tol}' 
                 f'Took {iterations} iterations')
           return True, next_point

        abs_dist = np.linalg.norm(current_point- next_point)
        if abs_dist <= param_tol:
            print( f'SUCCESS! Distance between two consecutive iterations iteration locations is {abs_dist}\n' 
                         f'which is less than the objactive tolerance {param_tol}\n' 
                         f'Took {iterations} iterations')
            return True, next_point

        if iterations == max_iter:
            #return False, f"FAIULRE! Did not converge in {iterations} iterations"
            return False, next_point

        iterations += 1
        current_point, current_val = next_point, next_val
    # The success/failure status should be printed toc onsole in human readable form describing the result (which convergence/failure, etc.).