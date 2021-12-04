# Authors: Hari Raval, Zongjun Liu, Aditi Memani, Joseph Zuccarelli             #
# Course: AC 207                                                                #
# File: test_forward_mode.py                                                    #
# Description: Test forward mode automatic differentiation by creating a        #
# testing suite using unittest                                                  #
#################################################################################

import unittest
import numpy as np
from ad_AHJZ.val_derv import val_derv
from ad_AHJZ.forward_mode import combine_vector_inputs, forward_mode


class forwardtest(unittest.TestCase):

    # test the combine_vector_inputs() function
    def test_combine_vector_inputs(self):
        # function to create the seed vector
        def seed(position):
            seed_vector = np.zeros(2)
            seed_vector[position] = 1
            return seed_vector

        # create val_derv objects for all inputs
        inputs = [1, 2]
        var_init = []
        for i, value in enumerate(inputs):
            var_init.append(val_derv(inputs[i], seed(i)))

        # create a test function and combine the vector inputs
        func = lambda x,y: np.array([x + y, x * y])
        res = func(*var_init)
        funct_val, funct_der = combine_vector_inputs(res, 2)

        # test if the array contents are almost equal to analytical values
        np.testing.assert_array_almost_equal(funct_val, np.array([3, 2]))
        np.testing.assert_array_almost_equal(funct_der, np.array([[1, 1], [2, 1]]))

    # test the get_function_value() and get_jacobian_value() of a univariate scalar function
    def test_univariate_scalar_f(self):
        # create a function, its analytical function value and derivative
        func = lambda x: 2 * x + x ** 2 + x.log() + 1 / x
        func_val = lambda x: 2 * x + x ** 2 + np.log(x) + 1 / x
        func_derv = lambda x: 2 + 2 * x + 1 / x - 1 / (x ** 2)

        # retrieve the package function value and derivative
        ad = forward_mode(2, func)
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()

        # test if array contents between our package and analytical solution are almost equal
        self.assertAlmostEqual(f_val, func_val(2))
        self.assertAlmostEqual(f_derv, func_derv(2))

    # test the get_function_value() and get_jacobian_value() of a multivariate scalar function
    def test_multivariate_scalar_f(self):
        val = np.array([1 / 2, 2, 3])
        # create a function, its analytical function value and derivative
        func = lambda x, y, z: x.arccos() + y.sin() * (x * y * z).tanh()
        func_val = lambda x, y, z: np.arccos(x) + np.sin(y) * np.tanh(x * y * z)
        func_derv = lambda x, y, z: (- 1 / np.sqrt(1 - x ** 2) + np.sin(y) * y * z / (np.cosh(x * y * z) ** 2),
                                     np.cos(y) * np.tanh(x * y * z) + np.sin(y) * x * z / (np.cosh(x * y * z) ** 2),
                                     np.sin(y) * x * y / (np.cosh(x * y * z) ** 2))

        # retrieve the package function value and derivative
        ad = forward_mode(val, func)
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()

        # test if array contents between our package and analytical solution are almost equal
        self.assertAlmostEqual(f_val, func_val(1 / 2, 2, 3))
        np.testing.assert_array_almost_equal(f_derv, func_derv(1 / 2, 2, 3))

    # test the get_function_value() and get_jacobian_value() of a multivariate vector function
    def test_multivariate_vector_f(self):
        val = np.array([1, 2, 3])
        # create a function, its analytical function value and derivative
        func = lambda x, y, z: (x.sin(), y.cos(), z.tan())
        func_val = lambda x, y, z: (np.sin(x), np.cos(y), np.tan(z))
        func_derv = lambda x, y, z: ([np.cos(x), 0, 0], [0, -np.sin(y), 0], [0, 0, 1 / np.cos(z) ** 2])

        # retrieve the package function value and derivative
        ad = forward_mode(val, func)
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()

        # test if array contents between our package and analytical solution are almost equal
        np.testing.assert_array_almost_equal(f_val, func_val(1, 2, 3))
        np.testing.assert_array_almost_equal(f_derv, func_derv(1, 2, 3))

    # test the get_function_value() and get_jacobian_value() of a multivariate vector function
    def test_univariate_vector_f(self):
        # create a function, its analytical function value and derivative
        func = lambda x: (x.sqrt(), x.arctan(), x.exp())
        func_val = lambda x: (np.sqrt(x), np.arctan(x), np.exp(x))
        func_derv = lambda x: ([1 / (2 * np.sqrt(x))], [1 / (1 + x ** 2)], [np.exp(x)])

        # retrieve the package function value and derivative
        ad = forward_mode(2, func)
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()

        # test if array contents between our package and analytical solution are almost equal
        np.testing.assert_array_almost_equal(f_val, func_val(2))
        np.testing.assert_array_almost_equal(f_derv, func_derv(2))
