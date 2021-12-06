# Authors: Hari Raval, Zongjun Liu, Aditi Memani, Joseph Zuccarelli             #
# Course: AC 207                                                                #
# File: test_optimizers.py                                                      #
# Description: Test the optimizer class functionality by creating a testing     #
# testing suite using unittest                                                  #
#################################################################################

import unittest
import numpy as np
from ad_AHJZ.optimizers import Optimizer

# create sample functions to find the minimum of through all unit test cases
x = 1
f_x = lambda x: (x + 1) ** 2
xy = np.array([0, 0])
f_xy = lambda x, y: (x - 1) ** 2 + (y + 1) ** 2


class optimizerstest(unittest.TestCase):

    # test univariate function for ADAM optimizer
    def test_univariate_ADAM(self):
        opt_time, val, curr_val = Optimizer.ADAM(x, f_x, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], -1, places=3)

    # test multivariate function for ADAM optimizer
    def test_multivariate_ADAM(self):
        opt_time, val, curr_val = Optimizer.ADAM(xy, f_xy, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], 1, places=3)
        self.assertAlmostEqual(curr_val[1], -1, places=3)

    # test univariate function for NADAM optimizer
    def test_univariate_NADAM(self):
        opt_time, val, curr_val = Optimizer.NADAM(x, f_x, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], -1, places=3)

    # test multivariate function for NADAM optimizer
    def test_multivariate_NADAM(self):
        opt_time, val, curr_val = Optimizer.NADAM(xy, f_xy, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], 1, places=3)
        self.assertAlmostEqual(curr_val[1], -1, places=3)

    # test univariate function for RMSprop optimizer
    def test_univariate_RMSprop(self):
        opt_time, val, curr_val = Optimizer.RMSprop(x, f_x, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], -1, places=3)

    # test multivariate function for RMSprop optimizer
    def test_multivariate_RMSprop(self):
        opt_time, val, curr_val = Optimizer.RMSprop(xy, f_xy, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], 1, places=3)
        self.assertAlmostEqual(curr_val[1], -1, places=3)

    # test univariate function for momentum optimizer
    def test_univariate_momentum(self):
        opt_time, val, curr_val = Optimizer.momentum(x, f_x, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], -1, places=3)

    # test multivariate function for momentum optimizer
    def test_multivariate_momentum(self):
        opt_time, val, curr_val = Optimizer.momentum(xy, f_xy, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], 1, places=3)
        self.assertAlmostEqual(curr_val[1], -1, places=3)

    # test univariate function for BFGS optimizer
    def test_univariate_BFGS(self):
        opt_time, val, curr_val = Optimizer.BFGS(x, f_x, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], -1, places=3)

    # test multivariate function for BFGS optimizer
    def test_multivariate_BFGS(self):
        opt_time, val, curr_val = Optimizer.BFGS(xy, f_xy, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], 1, places=3)
        self.assertAlmostEqual(curr_val[1], -1, places=3)

    # test univariate function for Broyden optimizer
    def test_univariate_broyden(self):
        opt_time, val, curr_val = Optimizer.broyden(x, f_x, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], -1, places=3)

    # test multivariate function for Broyden optimizer
    def test_multivariate_broyden(self):
        opt_time, val, curr_val = Optimizer.broyden(xy, f_xy, 1000)
        self.assertAlmostEqual(val, 0, places=3)
        self.assertAlmostEqual(curr_val[0], 1, places=3)
        self.assertAlmostEqual(curr_val[1], -1, places=3)
