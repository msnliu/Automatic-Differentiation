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
    ########################
    combine_vector_inputs(function_list, n_col)

    Parameters
    ----------
    function_list: list object of elementary functions
    n_col: integer object that represents the number of columns

    Returns
    -------
    funct_val: an array of values for each input function
    funct_der: an array of derivative values for each function

    Examples
    --------
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
        """
        ########################
        __init__(self, input_values, input_function)

        Constructor for the forward mode class.

        Parameters
        ----------
        input_values: a list of input value
        input_function: input function

        Returns
        -------
        None
        """
        self.inputs = input_values
        self.functions = input_function

    def get_function_value(self):
        """
        ########################
        get_function_value(self)

        

        Parameters
        ----------
        None

        Returns
        -------
        function value 

        Examples
        --------
        """
        return self.get_function_value_and_jacobian()[0]

    def get_jacobian(self):
        """
        ########################
        get_jacobian(self)

        Parameters
        ----------
        None

        Returns
        -------
        Jacobian matrix

        Examples
        --------
        """
        return self.get_function_value_and_jacobian()[1]

    def get_function_value_and_jacobian(self):
        """
        ########################
        get_function_value_and_jacobian(self)

        Parameters
        ----------
        None

        Returns
        -------
        value and derivative of the input function

        Examples
        --------
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
