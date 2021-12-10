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
        func = lambda x, y: np.array([x + y, x * y])
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

    # test the get_function_value() and get_jacobian_value() of an embedded univariate scalar function
    def test_embedded_univariate_scalar_f(self):
        # create a function, its analytical function value and derivative
        func = lambda x: (x.sin()).exp() - (x ** 0.5).cos() * ((x.cos() ** 2.0 + x ** 2.0) ** 0.5).sin()
        func_val = lambda x: np.exp(np.sin(x)) - np.cos(x ** 0.5) * np.sin((np.cos(x) ** 2.0 + x ** 2.0) ** 0.5)
        func_derv = lambda x: 0.5 * x ** (-0.5) * np.sin(x ** 0.5) * np.sin((x ** 2.0 + np.cos(x) ** 2.0) ** 0.5) - \
                              (1.0 * x ** 1.0 - 1.0 * np.sin(x) * np.cos(x) ** 1.0) * \
                              (x ** 2.0 + np.cos(x) ** 2.0) ** (-0.5) * np.cos(x ** 0.5) * \
                              np.cos((x ** 2.0 + np.cos(x) ** 2.0) ** 0.5) + np.exp(np.sin(x)) * np.cos(x)

        # retrieve the package function value and derivative
        ad = forward_mode(5.5, func)
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()

        # test if array contents between our package and analytical solution are almost equal
        np.testing.assert_array_almost_equal(f_val, func_val(5.5))
        np.testing.assert_array_almost_equal(f_derv, func_derv(5.5))

    # test the get_function_value() and get_jacobian_value() of an embedded multivariate scalar function
    def test_embedded_multivariate_vector_f(self):
        # create a function, its analytical function value and derivative
        val = np.array([1, 2, 3])
        func = lambda x, y, z: ((x + y).logistic(), (y * z).logistic(), (z / x).logistic())
        func_val = lambda x, y, z: (1 / (1 + np.exp(-(x + y))), 1 / (1 + np.exp(-(y * z))), 1 / (1 + np.exp(-(z / x))))
        func_derv = lambda x, y, z: (
            [np.exp(-x - y) / (np.exp(-x - y) + 1) ** 2, np.exp(-x - y) / (np.exp(-x - y) + 1) ** 2, 0],
            [0, z * np.exp(-y * z) / (1 + np.exp(-y * z)) ** 2, y * np.exp(-y * z) / (1 + np.exp(-y * z)) ** 2],
            [-z * np.exp(-z / x) / (x ** 2 * (1 + np.exp(-z / x)) ** 2), 0,
             np.exp(-z / x) / (x * (1 + np.exp(-z / x)) ** 2)])

        # retrieve the package function value and derivative
        ad = forward_mode(val, func)
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()

        # test if array contents between our package and analytical solution are almost equal
        np.testing.assert_array_almost_equal(f_val, func_val(1, 2, 3))
        np.testing.assert_array_almost_equal(f_derv, func_derv(1, 2, 3))

    # test a user-defined seed vector which is non-one scalar
    def test_seed_vector_not_one_scalar(self):
        # create a function, its analytical function value and derivative
        func = lambda x: (x ** 2).exp()
        ad = forward_mode(1, func, -2)
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()
        func_val = lambda x: np.exp(x ** 2)
        func_derv = lambda x: 2 * x * (-2) * np.exp(x ** 2)

        # test if array contents between our package and analytical solution are almost equal
        np.testing.assert_array_almost_equal(f_val, func_val(1))
        np.testing.assert_array_almost_equal(f_derv, func_derv(1))

    # test a user-defined seed vector which is non-one vector
    def test_seed_vector_not_one_vector(self):
        # create a function, its analytical function value and derivative
        val = np.array([1, 2, 3])
        func = lambda x, y, z: x / (y * z)
        ad = forward_mode(val, func, [-1, -2, 3])
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()
        func_val = lambda x, y, z: x / (y * z)
        func_derv = lambda x, y, z: (-1 / (y * z), 2 * x / (y ** 2 * z), -3 * x / (y * z ** 2))

        # test if array contents between our package and analytical solution are almost equal
        np.testing.assert_array_almost_equal(f_val, func_val(1, 2, 3))
        np.testing.assert_array_almost_equal(f_derv, func_derv(1, 2, 3))

    # test a user-defined seed vector which is an array of length one
    def test_seed_vector_not_one_invalid(self):
        # create a function, its analytical function value and derivative
        val = 1
        func = lambda x: x.sqrt()
        ad = forward_mode(val, func, [-1])
        f_val = ad.get_function_value()
        f_derv = ad.get_jacobian()
        func_val = lambda x: np.sqrt(x)
        func_derv = lambda x: -1 / 2 * x ** (-1 / 2)

        # test if array contents between our package and analytical solution are almost equal
        np.testing.assert_array_almost_equal(f_val, func_val(1))
        np.testing.assert_array_almost_equal(f_derv, func_derv(1))

    # test a user-defined seed vector which is an incorrect length (seed longer than input variables)
    def test_seed_vector_incorrect_long_length(self):
        # create a function, its analytical function value and derivative
        val = 1
        func = lambda x: x.sqrt()
        ad = forward_mode(val, func, [-1, 2])

        with self.assertRaises(ValueError) as e:
            ad.get_function_value()
        self.assertEqual("ERROR: Inputted seed vector length and number of variable mismatch", str(e.exception))

    # test a user-defined seed vector which is an incorrect length (seed shorter than input variables)
    def test_seed_vector_incorrect_short_length(self):
        # create a function, its analytical function value and derivative
        val = np.array([1, 2, 3])
        func = lambda x, y, z: x / (y * z)
        ad = forward_mode(val, func, [-1, 2])

        with self.assertRaises(ValueError) as e:
            ad.get_function_value_and_jacobian()
        self.assertEqual("ERROR: Inputted seed vector length and number of variable mismatch", str(e.exception))
