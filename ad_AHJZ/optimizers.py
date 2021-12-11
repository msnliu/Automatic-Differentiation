# Authors: Hari Raval, Zongjun Liu, Aditi Memani, Joseph Zuccarelli             #
# Course: AC 207                                                                #
# File: optimzers.py                                                            #
# Description: Create six different optimization techniques that leverage       #
# forward mode automatic differentiation, enabling a user to find the minimum   #
# value of a function, location of the minimum value, and wall clock time to    #
# find these values.                                                            #
#################################################################################

import numpy as np
from ad_AHJZ.forward_mode import forward_mode
import time


class Optimizer:
    """
    A class containing six different optimizer methods that leverage forward mode
    automatic differentiation, enabling a user to to find the minimum value of a function,
    location of the minimum value, and wall clock time to find these values

    Examples
    --------
    # sample use case of instantiating and using an adam optimizer
    >>> x = 1
    >>> opt = Optimizer()
    >>> f_x = lambda x: x** 2
    >>> print(opt.ADAM(x, f_x, 1000))
    (0.058393001556396484, 3.2900616291354705e-42, array([-1.8138527e-21]))

    # sample use case of instantiating and using a momentum optimizer
    >>> x = 2
    >>> opt = Optimizer()
    >>> f_x = lambda x: x** 4
    >>> print(opt.momentum(x, f_x, 1000))
    (0.04654884338378906, 2.0278841795288425e-05, array([-0.06710591]))

    """

    @staticmethod
    def momentum(x, f_x, num_iter, alpha=0.01, beta=.9):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        f_x: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform
        alpha: learning rate for the gradiant descent (default 0.01)
        beta: exponential decay (default 0.9)

        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> f_x = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.momentum(x, f_x, 1000)
        (0.24898195266723633, 0.26172998379097046, array([0.94233316]))

        >>> x = np.array([1, -1])
        >>> f_x = lambda x, y:x**3 + y**2
        >>> Optimizer.momentum(x, f_x, 1000)
        (0.08585095405578613, 2.7605629377339922e-05, array([ 3.02226506e-02, -3.30135704e-12]))

        >>> x = 2
        >>> f_x = lambda x: (x - 1)**2 + 5
        >>> Optimizer.momentum(x, f_x, 1000)
        (0.06420779228210449, 5.0, array([1.]))

        """

        # start the timer
        start = time.time()
        # decay value must be great than or equal to 0 and less than 1
        if 0 <= beta < 1:
            mt, curr_val = 0, x
            fm = forward_mode(x, f_x)
            val, x_der = fm.get_function_value(), fm.get_jacobian()

            # perform momentum optimization for the number of iterations specified
            for t in range(1, num_iter + 1):
                mt = beta * mt + (1 - beta) * x_der
                variation = alpha * mt
                # compute the new variation to update the current x location
                curr_val = curr_val - variation
                # recalculate the function value and derivative at the updated value
                fm = forward_mode(curr_val, f_x)
                val, x_der = fm.get_function_value(), fm.get_jacobian()
        # raise the appropriate error for beta value not within 0 to 1
        else:
            raise ValueError("Beta Values must be within the range of [0,1)")

        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        return opt_time, val, curr_val

    @staticmethod
    def ADAM(x, f_x, num_iter, alpha=0.01, beta1=.9, beta2=.999, epsilon=1e-8):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        f_x: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform
        alpha: learning rate for the gradiant descent (default 0.01)
        beta1: exponential decay of the first moment (default 0.90)
        beta2: exponential decay of the second moment (default 0.999)
        epsilon: denominator value to assure that ZeroDivisionError is not raised (default 1e-8)

        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> f_x = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.ADAM(x, f_x, 1000)
        (0.19765210151672363, 0.26172998379097046, array([0.94233316]))

        >>> x = np.array([1,-1])
        >>> f_x = lambda x,y:x**3 + y**2
        >>> Optimizer.ADAM(x, f_x, 1000)
        (0.09849190711975098, 6.03886825409073e-06, array([1.82103595e-02, 1.81385270e-21]))

        >>> x = 2
        >>> f_x = lambda x: (x-1)**2 + 5
        >>> Optimizer.ADAM(x, f_x, 1000)
        (0.08600378036499023, 5.0, array([1.]))

        """

        # start the timer
        start = time.time()

        # decay value must be great than or equal to 0 and less than 1
        if 0 <= beta1 < 1:
            if 0 <= beta2 < 1:
                # initialize moment and direction variables
                mt, vt = 0, 0
                curr_val = x
                fm = forward_mode(x, f_x)
                val, x_der = fm.get_function_value(), fm.get_jacobian()

                # perform ADAM optimization for the number of iterations specified
                for t in range(1, num_iter + 1):
                    mt = beta1 * mt + (1 - beta1) * x_der
                    vt = beta2 * vt + (1 - beta2) * x_der ** 2
                    mhat = mt / (1 - beta1 ** t)
                    vhat = vt / (1 - beta2 ** t)
                    # compute the new variation to update the current x location
                    variation = alpha * mhat / (np.sqrt(vhat) + epsilon)
                    curr_val = curr_val - variation
                    # recalculate the function value and derivative at the updated value
                    fm = forward_mode(curr_val, f_x)
                    val, x_der = fm.get_function_value(), fm.get_jacobian()
            # raise the appropriate error for beta 1 value not within 0 to 1
            else:
                raise ValueError("Beta Values must be within the range of [0,1)")
        # raise the appropriate error for beta 2 value not within 0 to 1
        else:
            raise ValueError("Beta Values must be within the range of [0,1)")

        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        return opt_time, val, curr_val

    @staticmethod
    def NADAM(x, f_x, num_iter, alpha=0.01, beta1=.9, beta2=.999, epsilon=1e-8):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        f_x: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform
        alpha: learning rate for the gradiant descent (default 0.01)
        beta1: exponential decay of the first moment (default 0.90)
        beta2: exponential decay of the second moment (default 0.999)
        epsilon: denominator value to assure that ZeroDivisionError is not raised (default 1e-8)

        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> f_x = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.NADAM(x, f_x, 1000)
        (0.17513608932495117, 0.26172998379097046, array([0.94233316]))

        >>> x = np.array([1,-1])
        >>> f_x = lambda x, y:x**3 + y**2
        >>> Optimizer.NADAM(x, f_x, 1000)
        (0.09782099723815918, 6.246798742956698e-06, array([ 1.8417012e-02, -2.7649170e-21]))

        >>> x = 2
        >>> f_x = lambda x: (x-1)**2 + 5
        >>> Optimizer.NADAM(x, f_x, 1000)
        (0.07419586181640625, 5.0, array([1.]))

        """

        # start the timer
        start = time.time()
        # decay value must be great than or equal to 0 and less than 1
        if 0 <= beta1 < 1:
            if 0 <= beta2 < 1:
                # initialize moment and direction variables
                mt, vt = 0, 0
                curr_val = x
                fm = forward_mode(x, f_x)
                val, x_der = fm.get_function_value(), fm.get_jacobian()

                # perform NADAM optimization for the number of iterations specified
                for t in range(1, num_iter + 1):
                    mt = beta1 * mt + (1 - beta1) * x_der
                    vt = beta2 * vt + (1 - beta2) * x_der ** 2
                    mhat = mt / (1 - beta1 ** t)
                    vhat = vt / (1 - beta2 ** t)
                    mvar = beta1 * mhat + (1 - beta1) * x_der
                    # compute the new variation to update the current x location (note: epsilon is under square root)
                    variation = alpha * mvar / (np.sqrt(vhat + epsilon))
                    curr_val = curr_val - variation
                    # recalculate the function value and derivative at the updated value
                    fm = forward_mode(curr_val, f_x)
                    val, x_der = fm.get_function_value(), fm.get_jacobian()
            # raise the appropriate error for beta 1 value not within 0 to 1
            else:
                raise ValueError("Beta Values must be within the range of [0,1)")
        # raise the appropriate error for beta 2 value not within 0 to 1
        else:
            raise ValueError("Beta Values must be within the range of [0,1)")

        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        return opt_time, val, curr_val

    @staticmethod
    def RMSprop(x, f_x, num_iter, alpha=0.01, beta=.9, epsilon=1e-8):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        f_x: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform
        alpha: learning rate for the gradiant descent (default 0.01)
        beta: exponential decay (default 0.9)
        epsilon: denominator value to assure that ZeroDivisionError is not raised (default 1e-8)

        Returns
        -------
        opt_time: The time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> f_x = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.RMSprop(x, f_x, 1000)
        (0.3034090995788574, 0.2618028370373199, array([0.93730206]))

        >>> x = np.array([1,-1])
        >>> f_x = lambda x, y:x**3 + y**2
        >>> Optimizer.RMSprop(x, f_x, 1000)
        (0.08888602256774902, 2.4997500000081948e-05, array([ 4.34424616e-06, -4.99974999e-03]))

        >>> x = 2
        >>> f_x = lambda x: (x - 1)**2 + 5
        >>> Optimizer.RMSprop(x, f_x, 1000)
        (0.07326507568359375, 5.0, array([1.]))

        """

        # start the timer
        start = time.time()
        # decay value must be great than or equal to 0 and less than 1
        if 0 <= beta < 1:
            vt, curr_val = 0, x
            fm = forward_mode(x, f_x)
            val, x_der = fm.get_function_value(), fm.get_jacobian()

            # perform RMSprop optimization for the number of iterations specified
            for t in range(1, num_iter + 1):
                vt = beta * vt + (1 - beta) * x_der ** 2
                variation = alpha * x_der / (np.sqrt(vt + epsilon))
                # compute the new variation to update the current x location
                curr_val = curr_val - variation
                # recalculate the function value and derivative at the updated value
                fm = forward_mode(curr_val, f_x)
                val, x_der = fm.get_function_value(), fm.get_jacobian()
        # raise the appropriate error for beta value not within 0 to 1
        else:
            raise ValueError("Beta Values must be within the range of [0,1)")

        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        return opt_time, val, curr_val

    @staticmethod
    def broyden(x, f_x, num_iter, alpha=0.01):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        f_x: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform
        alpha: learning rate for the gradiant descent (default 0.01)

        Returns
        -------
        opt_time: the time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> f_x = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.broyden(x, f_x, 1000)
        (0.2750861644744873, 0.2617299838095016, array([0.94233569]))

        >>> x = np.array([1, -1])
        >>> f_x = lambda x, y:x**3 + y**2
        >>> Optimizer.broyden(x, f_x, 1000)
        (0.1135101318359375, 3.060640040173604e-07, array([ 6.72566138e-03, -4.28010827e-05]))

        >>> x = 2
        >>> f_x = lambda x: (x - 1)**2 + 5
        >>> Optimizer.broyden(x, f_x, 1000)
        broyden: (0.0851430892944336, 5.000000001826295, array([1.00004274]))

        """

        # start the timer
        start = time.time()
        if np.isscalar(x):
            n = 1
        else:
            n = len(x)
        curr_val = x
        # start with the hessian as the identity matrix
        curr_hessian = np.eye(n)
        fm = forward_mode(x, f_x)
        val, x_der = fm.get_function_value(), fm.get_jacobian()

        # perform broyden optimization for the number of iterations specified
        for t in range(1, num_iter + 1):
            # compute the new variation to update the current x location
            variation = -alpha * curr_hessian @ x_der
            curr_val = curr_val + variation
            # recalculate the function value and derivative at the updated value
            fm = forward_mode(curr_val, f_x)
            val, x_der1 = fm.get_function_value(), fm.get_jacobian()
            yk = (x_der1 - x_der).reshape(-1, 1)
            xk = variation.reshape(-1, 1)
            # compute the amount the hessian will change by
            broyden_numerator = (xk - curr_hessian @ yk) @ xk.T @ curr_hessian
            broyden_denom = xk.T @ curr_hessian @ yk
            # update the hessian matrix per iteration (second-order method)
            curr_hessian = curr_hessian + broyden_numerator / broyden_denom
            x_der = x_der1

        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        return opt_time, val, curr_val

    @staticmethod
    def BFGS(x, f_x, num_iter, alpha=0.01):
        """
        Parameters
        ----------
        x: the variable input (can be in either scalar or vector form)
        f_x: the function you would like to obtain the minimum for
        num_iter: the number of interations to perform
        alpha: learning rate for the gradiant descent (default 0.01)

        Returns
        -------
        opt_time: the time it takes to run the optimizer in seconds
        val: the position of the minimum value
        curr_val: the minimum value (can be in either scalar or vector form)

        Examples
        --------
        >>> x = 1
        >>> f_x = lambda x: (-1 * x.log()) + (x.exp() * x**4) / 10
        >>> Optimizer.BFGS(x, f_x, 1000)
        (0.2875189781188965, 0.2617299838095016, array([0.94233569]))

        >>> x = np.array([1, -1])
        >>> f_x = lambda x, y:x**3 + y**2
        >>> Optimizer.BFGS(x, f_x, 1000)
        (0.12035322189331055, 2.864368146437936e-07, array([ 6.57872856e-03, -4.13716985e-05]))

        >>> x = 2
        >>> f_x = lambda x: (x - 1)**2 + 5
        >>> Optimizer.BFGS(x, f_x, 1000)
        ((0.08925986289978027, 5.000000001826295, array([1.00004274]))

        """

        # start the timer
        start = time.time()
        if np.isscalar(x):
            n = 1
        else:
            n = len(x)
        curr_val = x
        # start with the hessian as the identity matrix
        curr_hessian = np.eye(n)
        fm = forward_mode(x, f_x)
        val, x_der = fm.get_function_value(), fm.get_jacobian()

        # perform BFGS optimization for the number of iterations specified
        for t in range(1, num_iter + 1):
            # compute the new variation to update the current x location
            variation = -alpha * curr_hessian @ x_der
            curr_val = curr_val + variation
            # recalculate the function value and derivative at the updated value
            fm = forward_mode(curr_val, f_x)
            val, x_der1 = fm.get_function_value(), fm.get_jacobian()
            I_mat = np.eye(n)
            y_update, x_update = (x_der1 - x_der).reshape(-1, 1), variation.reshape(-1, 1)
            denom = y_update.T @ x_update
            # compute each matrix individually to determine the amount the hessian will change by
            first_term, sec_term, third_term = (I_mat - x_update @ y_update.T / denom), \
                                               (I_mat - y_update @ x_update.T / denom), \
                                               x_update @ x_update.T / denom

            # update the hessian matrix per iteration (second-order method)
            curr_hessian = first_term @ curr_hessian @ sec_term + third_term
            x_der = x_der1

        # end the timer and compute wall clock time
        end = time.time()
        opt_time = end - start
        return opt_time, val, curr_val
