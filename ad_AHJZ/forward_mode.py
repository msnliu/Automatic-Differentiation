# Authors: Hari Raval, Zongjun Liu, Aditi Memani, Joseph Zuccarelli             #
# Course: AC 207                                                                #
# File: forward_mode.py                                                         #
# Description: Perform forward mode automatic differentiation, enabling a user  #
# to output just the function values, just the derivative values, or both the   #
# function and derviative values in a tuple                                     #
#################################################################################

import numpy as np
from ad_AHJZ.val_derv import val_derv


def combine_vector_inputs(function_list, n_col):
    """
    Parameters
    ----------
    function_list: list object of functions
    n_col: integer object that represents the number of columns

    Returns
    -------
    funct_val: an array of values for each input function
    funct_der: an array of derivative values for each function

    Examples
    --------
    >>> def seed(position):
    >>>     seed_vector = np.zeros(2)
    >>>     seed_vector[position] = 1
    >>>     return seed_vector

    >>> inputs = [1, 2]
    >>> var_init = []
    >>> for i, value in enumerate(inputs):
    >>>     var_init.append(val_derv(inputs[i], seed(i)))

    >>> func = lambda x,y: np.array([x + y, x * y])
    >>> res = func(*var_init)
    >>> funct_val, funct_der = combine_vector_inputs(res, 2)
    >>> print(funct_val)
    [3. 2.]
    >>> print(funct_der)
    [[1. 1.]
    [2. 1.]]
    """

    # determine the number of functions inputted
    number_functions = len(function_list)

    # arrays to store the function and derivative values
    funct_val = np.empty(number_functions)
    funct_der = np.empty([number_functions, n_col])

    # compute the function and derivative value for each of the functions
    for i, funct in enumerate(function_list):
        funct_val[i] = funct.val
        funct_der[i] = funct.derv

    return funct_val, funct_der

class forward_mode:

    def __init__(self, input_values, input_function):
        self.inputs = input_values
        self.functions = input_function

    def get_function_value(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        function value

        Examples
        --------
        # get univariate scalar function value
        >>> func = lambda x: x
        >>> fm = forward_mode(1, func)
        >>> fm.get_function_value()
        1

        # get univariate vector function value
        >>> func = lambda x: (x, 2*x, x**2)
        >>> fm = forward_mode(1, func)
        >>> fm.get_function_value()
        array([1., 2., 1.])

        # get multivariate scalar function value
        >>> func = lambda x, y: x + y
        >>> fm = forward_mode(np.array([1, 1]), func)
        >>> fm.get_function_value()
        2

        # get multivariate vector function value
        >>> func = lambda x, y: (x.log() + y, x + y)
        >>> fm = forward_mode(np.array([1, 1]), func)
        >>> fm.get_function_value()
        array([1., 2.])
        """

        return self.get_function_value_and_jacobian()[0]

    def get_jacobian(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        jacobian

        Examples
        --------
        # get univariate scalar function jacobian
        >>> func = lambda x: x
        >>> fm = forward_mode(1, func)
        >>> fm.get_jacobian()
        array([1.])

        # get univariate vector function jacobian
        >>> func = lambda x: (x, 2*x, x**2)
        >>> fm = forward_mode(1, func)
        >>> fm.get_jacobian()
        array([[1.],
               [2.],
               [2.]])

        # get multivariate scalar function jacobian
        >>> func = lambda x, y: x + y
        >>> fm = forward_mode(np.array([1, 1]), func)
        >>> fm.get_jacobian()
        array([1., 1.])

        # get multivariate vector function jacobian
        >>> func = lambda x, y: (x.log() + y, x + y)
        >>> fm = forward_mode(np.array([1, 1]), func)
        >>> fm.get_jacobian()
        array([[1., 1.],
               [1., 1.]])
        """

        return self.get_function_value_and_jacobian()[1]

    def get_function_value_and_jacobian(self):
        """
        Parameters
        ----------
        None

        Returns
        -------
        value and jacobian of the input function

        Examples
        --------
        # get univariate scalar function value and jacobian
        >>> func = lambda x: x
        >>> fm = forward_mode(1, func)
        >>> fm.get_function_value_and_jacobian()
        (1, array([1.]))

        # get univariate vector function value and jacobian
        >>> func = lambda x: (x, 2*x, x**2)
        >>> fm = forward_mode(1, func)
        >>> fm.get_function_value_and_jacobian()
        (array([1., 2., 1.]), array([[1.],
                                     [2.],
                                     [2.]]))

        # get multivariate scalar function value and jacobian
        >>> func = lambda x, y: x + y
        >>> fm = forward_mode(np.array([1, 1]), func)
        >>> fm.get_function_value_and_jacobian()
        (2, array([1., 1.]))

        # get multivariate vector function value and jacobian
        >>> func = lambda x, y: (x.log() + y, x + y)
        >>> fm = forward_mode(np.array([1, 1]), func)
        >>> fm.get_function_value_and_jacobian()
        (array([1., 2.]), array([[1., 1.],
                                 [1., 1.]]))
        """

        # number of input values = number of variables = number of partial derivatives we want to compute
        try:
            # find the number of terms to evaluate
            number_res = len(self.inputs)
        # handle the case of having only a single input value: convert the scalar value into a list
        except TypeError:
            number_res = 1
            self.inputs = np.array([self.inputs])

        # initialize the number of partial derivatives
        var_init = [0] * number_res

        # compute the seed vector based on the current position
        def seed(position):
            seed_vector = np.zeros(number_res)
            seed_vector[position] = 1
            return seed_vector

        # initialize each variable into a val_derv() object
        for i, value in enumerate(self.inputs):
            var_init[i] = val_derv(self.inputs[i], seed(i))

        res = self.functions(*var_init)

        try:
            # return the results for the scalar case
            return res.val, res.derv

        except AttributeError:
            # return the results for the vector case
            return combine_vector_inputs(res, number_res)

