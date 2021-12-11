# Authors: Hari Raval, Zongjun Liu, Aditi Memani, Joseph Zuccarelli             #
# Course: AC 207                                                                #
# File: test_val_derv.py                                                        #
# Description: Test the val_derv object functionality by creating a testing     #
# testing suite using unittest                                                  #
#################################################################################
import unittest
import numpy as np

from ad_AHJZ.val_derv import val_derv, var_type

# create multiple val_derv objects to use through all unit test cases
var1 = val_derv(1, -1)
var2 = val_derv(2.2, 0)
var3 = val_derv(-1, np.pi)
var4 = val_derv(-5.3, 124)
var5 = val_derv(2, 6.036)
var6 = val_derv(0, 1)
var7 = val_derv(-3, -100000)
var8 = val_derv(3, 100000)
var9 = val_derv(0.5, -2)
var10 = val_derv(0, 1)
var11 = val_derv(1, 1)

x = val_derv(1, np.array([-1, 2]))
y = val_derv(-1, np.array([np.pi, 6.033]))
z = val_derv(0, np.array([1, -2]))
w = val_derv(0.5, np.array([10000, -10000]))
t = val_derv(-2, np.array([1, -2]))
p = val_derv(2, np.array([1, -2]))


class Val_Derv_Elem_Test(unittest.TestCase):

    # test scalar addition operation
    def test_scalar_add(self):
        add_scalar = var1 + var2
        self.assertAlmostEqual(3.2, add_scalar.val)
        self.assertAlmostEqual(-1, add_scalar.derv)
        add_scalar_const = var1 + 2
        self.assertAlmostEqual(3, add_scalar_const.val)
        self.assertAlmostEqual(-1, add_scalar_const.derv)
        add_scalar_rev = 2 + var1
        self.assertAlmostEqual(3, add_scalar_rev.val)
        self.assertAlmostEqual(-1, add_scalar_rev.derv)

    # test vector addition operation
    def test_vector_add(self):
        add_vector = x + y
        self.assertAlmostEqual(0, add_vector.val)
        np.testing.assert_array_almost_equal(np.array([-1 + np.pi, 2 + 6.033]), add_vector.derv)
        add_vector_const = x + 2
        self.assertAlmostEqual(3, add_vector_const.val)
        np.testing.assert_array_almost_equal(np.array([-1, 2]), add_vector_const.derv)
        add_vector_rev = 2 + x
        self.assertAlmostEqual(3, add_vector_rev.val)
        np.testing.assert_array_almost_equal(np.array([-1, 2]), add_vector_rev.derv)

    # test scalar subtraction operation
    def test_scalar_sub(self):
        sub_scalar = var1 - var2
        self.assertAlmostEqual(-1.2, sub_scalar.val)
        self.assertAlmostEqual(-1, sub_scalar.derv)
        sub_scalar_const = var1 - 2
        self.assertAlmostEqual(-1, sub_scalar_const.val)
        self.assertAlmostEqual(-1, sub_scalar_const.derv)
        sub_scalar_rev = 2 - var1
        self.assertAlmostEqual(1, sub_scalar_rev.val)
        self.assertAlmostEqual(1, sub_scalar_rev.derv)

    # test vector subtraction operation
    def test_vector_sub(self):
        sub_vector = x - y
        self.assertAlmostEqual(2, sub_vector.val)
        np.testing.assert_array_almost_equal(np.array([-1 - np.pi, 2 - 6.033]), sub_vector.derv)
        sub_vector_const = x - 2
        self.assertAlmostEqual(-1, sub_vector_const.val)
        np.testing.assert_array_almost_equal(np.array([-1, 2]), sub_vector_const.derv)
        sub_vector_rev = 2 - x
        self.assertAlmostEqual(1, sub_vector_rev.val)
        np.testing.assert_array_almost_equal(np.array([1, -2]), sub_vector_rev.derv)

    # test scalar multiplication operation
    def test_scalar_mul(self):
        prod_scalar = var1 * var3
        self.assertAlmostEqual(-1, prod_scalar.val)
        self.assertAlmostEqual(np.pi + 1, prod_scalar.derv)
        prod_scalar_const = var1 * 2
        self.assertAlmostEqual(2, prod_scalar_const.val)
        self.assertAlmostEqual(-2, prod_scalar_const.derv)
        pro_scalar_rev = 2 * var1
        self.assertAlmostEqual(2, pro_scalar_rev.val)
        self.assertAlmostEqual(-2, pro_scalar_rev.derv)

    # test vector multiplication operation
    def test_vector_mul(self):
        prod_vector = x * y
        self.assertAlmostEqual(-1, prod_vector.val)
        np.testing.assert_array_almost_equal(np.array([1 + np.pi, -2 + 6.033]), prod_vector.derv)
        prod_vector_const = x * 2
        self.assertAlmostEqual(2, prod_vector_const.val)
        np.testing.assert_array_almost_equal(np.array([-2, 4]), prod_vector_const.derv)
        prod_vector_rev = 2 * x
        self.assertAlmostEqual(2, prod_vector_rev.val)
        np.testing.assert_array_almost_equal(np.array([-2, 4]), prod_vector_rev.derv)

    # test scalar division operation
    def test_scalar_trudiv(self):
        div_scalar = var1 / var5
        self.assertAlmostEqual(0.5, div_scalar.val)
        self.assertAlmostEqual(-8.036 / 4, div_scalar.derv)
        div_scalar_const = var1 / 2
        self.assertAlmostEqual(0.5, div_scalar_const.val)
        self.assertAlmostEqual(-0.5, div_scalar_const.derv)
        div_scalar_rev = 2 / var1
        self.assertAlmostEqual(2, div_scalar_rev.val)
        self.assertAlmostEqual(2, div_scalar_rev.derv)

    # test vector division operation
    def test_vector_trudiv(self):
        div_vector = x / y
        self.assertAlmostEqual(-1, div_vector.val)
        np.testing.assert_array_almost_equal(np.array([1 - np.pi, -2 - 6.033]), div_vector.derv)
        div_vector_const = x / 2
        self.assertAlmostEqual(0.5, div_vector_const.val)
        np.testing.assert_array_almost_equal(np.array([-1 / 2, 1]), div_vector_const.derv)
        div_vector_rev = 2 / x
        self.assertAlmostEqual(2, div_vector_rev.val)
        np.testing.assert_array_almost_equal(np.array([2, -4]), div_vector_rev.derv)

    # test error handling in scalar division operation
    def test_scalar_trudiv_zeroError(self):
        with self.assertRaises(ZeroDivisionError) as e:
            var1 / 0
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

        with self.assertRaises(ZeroDivisionError) as e:
            var1 / var6
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

        with self.assertRaises(ZeroDivisionError) as e:
            3 / var6
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

    # test error handling in vector division operation
    def test_vector_trudiv_zeroError(self):
        with self.assertRaises(ZeroDivisionError) as e:
            x / 0
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

        with self.assertRaises(ZeroDivisionError) as e:
            x / z
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

        with self.assertRaises(ZeroDivisionError) as e:
            3 / z
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

    # test scalar negation operation
    def test_scalar_neg(self):
        neg_scalar = -var1
        self.assertAlmostEqual(-1, neg_scalar.val)
        self.assertAlmostEqual(1, neg_scalar.derv)
        neg_scalar_const = -var2
        self.assertAlmostEqual(-2.2, neg_scalar_const.val)
        self.assertAlmostEqual(0, neg_scalar_const.derv)

    # test vector negation operation
    def test_vector_neg(self):
        neg_vector = -x
        self.assertAlmostEqual(-1, neg_vector.val)
        np.testing.assert_array_almost_equal(np.array([1, -2]), neg_vector.derv)

    # test scalar sine operation
    def test_sin_scalar(self):
        sin_scalar = var5.sin()
        self.assertAlmostEqual(np.sin(2), sin_scalar.val)
        self.assertAlmostEqual(np.cos(2) * 6.036, sin_scalar.derv)
        sin_scalar_const = var2.sin()
        self.assertAlmostEqual(np.sin(2.2), sin_scalar_const.val)
        self.assertAlmostEqual(0, sin_scalar_const.derv)

    # test vector sine operation
    def test_sin_vector(self):
        sin_vector = x.sin()
        self.assertAlmostEqual(np.sin(1), sin_vector.val)
        np.testing.assert_array_almost_equal(np.cos(1) * np.array([-1, 2]), sin_vector.derv)

    # test scalar cosine operation
    def test_cos_scalar(self):
        cos_scalar = var5.cos()
        self.assertAlmostEqual(np.cos(2), cos_scalar.val)
        self.assertAlmostEqual(-np.sin(2) * 6.036, cos_scalar.derv)
        cos_scalar_const = var2.cos()
        self.assertAlmostEqual(np.cos(2.2), cos_scalar_const.val)
        self.assertAlmostEqual(0, cos_scalar_const.derv)

    # test vector cosine operation
    def test_cos_vector(self):
        cos_vector = x.cos()
        self.assertAlmostEqual(np.cos(1), cos_vector.val)
        np.testing.assert_array_almost_equal(-np.sin(1) * np.array([-1, 2]), cos_vector.derv)

    # test scalar tangent operation
    def test_tan_scalar(self):
        tan_scalar = var5.tan()
        self.assertAlmostEqual(np.tan(2), tan_scalar.val)
        self.assertAlmostEqual(6.036 / (np.cos(2) ** 2), tan_scalar.derv)
        tan_scalar_const = var2.tan()
        self.assertAlmostEqual(np.tan(2.2), tan_scalar_const.val)
        self.assertAlmostEqual(0, tan_scalar_const.derv)

    # test vector tangent operation
    def test_tan_vector(self):
        tan_vector = x.tan()
        self.assertAlmostEqual(np.tan(1), tan_vector.val)
        np.testing.assert_array_almost_equal(np.array([-1, 2]) / (np.cos(1) ** 2), tan_vector.derv)

    # test erorr handling in scalar tangent operation
    def test_tan_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = val_derv(3 * np.pi / 2, -1)
            var.tan()
        self.assertEqual("ERROR: Input to tan should not be an odd mutiple of pi/2", str(e.exception))

    # test erorr handling in vector tangent operation
    def test_tan_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = val_derv(3 * np.pi / 2, np.array([1, -1]))
            var.tan()
        self.assertEqual("ERROR: Input to tan should not be an odd mutiple of pi/2", str(e.exception))

    # test scalar hyperbolic sine operation
    def test_sinh_scalar(self):
        sinh_scalar = var5.sinh()
        self.assertAlmostEqual(np.sinh(2), sinh_scalar.val)
        self.assertAlmostEqual(np.cosh(2) * 6.036, sinh_scalar.derv)
        sinh_scalar_const = var2.sinh()
        self.assertAlmostEqual(np.sinh(2.2), sinh_scalar_const.val)
        self.assertAlmostEqual(0, sinh_scalar_const.derv)

    # test vector hyperbolic sine operation
    def test_sinh_vector(self):
        sinh_vector = x.sinh()
        self.assertAlmostEqual(np.sinh(1), sinh_vector.val)
        np.testing.assert_array_almost_equal(np.cosh(1) * np.array([-1, 2]), sinh_vector.derv)

    # test scalar hyperbolic cosine operation
    def test_cosh_scalar(self):
        cosh_scalar = var5.cosh()
        self.assertAlmostEqual(np.cosh(2), cosh_scalar.val)
        self.assertAlmostEqual(np.sinh(2) * 6.036, cosh_scalar.derv)
        cosh_scalar_const = var2.cosh()
        self.assertAlmostEqual(np.cosh(2.2), cosh_scalar_const.val)
        self.assertAlmostEqual(0, cosh_scalar_const.derv)

    # test vector hyperbolic cosine operation
    def test_cosh_vector(self):
        cosh_vector = x.cosh()
        self.assertAlmostEqual(np.cosh(1), cosh_vector.val)
        np.testing.assert_array_almost_equal(np.sinh(1) * np.array([-1, 2]), cosh_vector.derv)

    # test scalar hyperbolic tangent operation
    def test_tanh_scalar(self):
        tanh_scalar = var5.tanh()
        self.assertAlmostEqual(np.tanh(2), tanh_scalar.val)
        self.assertAlmostEqual((1 - np.tanh(2) ** 2) * 6.036, tanh_scalar.derv)
        tanh_scalar_const = var2.tanh()
        self.assertAlmostEqual(np.tanh(2.2), tanh_scalar_const.val)
        self.assertAlmostEqual(0, tanh_scalar_const.derv)

    # test vector hyperbolic tangent operation
    def test_tanh_vector(self):
        tanh_vector = x.tanh()
        self.assertAlmostEqual(np.tanh(1), tanh_vector.val)
        np.testing.assert_array_almost_equal((1 - np.tanh(1) ** 2) * np.array([-1, 2]), tanh_vector.derv)

    # test scalar logarithmic operation
    def test_log_scalar(self):
        log_scalar = var5.log()
        self.assertAlmostEqual(np.log(2), log_scalar.val)
        self.assertAlmostEqual(6.036 / 2, log_scalar.derv)
        log_scalar_base10 = var5.log(10)
        self.assertAlmostEqual(np.log(2) / np.log(10), log_scalar_base10.val)
        self.assertAlmostEqual((1 / (2 * np.log(10))) * 6.036, log_scalar_base10.derv)

    # test vector logarithmic operation
    def test_log_vector(self):
        log_vector = x.log()
        self.assertAlmostEqual(np.log(1), log_vector.val)
        np.testing.assert_array_almost_equal(1 * np.array([-1, 2]), log_vector.derv)

    # test error handling in scalar logarithmic operation
    def test_log_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var3.log()
        self.assertEqual("ERROR: Value for log should be greater than 0", str(e.exception))
        with self.assertRaises(ValueError) as e:
            var5.log(1)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))
        with self.assertRaises(ValueError) as e:
            var5.log(0)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))
        with self.assertRaises(ValueError) as e:
            var5.log(-1)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))

    # test error handling in vector logarithmic operation
    def test_log_vector_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            y.log()
        self.assertEqual("ERROR: Value for log should be greater than 0", str(e.exception))
        with self.assertRaises(ValueError) as e:
            x.log(1)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))
        with self.assertRaises(ValueError) as e:
            x.log(0)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))
        with self.assertRaises(ValueError) as e:
            x.log(-1)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))

    # test scalar exponential operation
    def test_exp_scalar(self):
        exp_scalar = var1.exp()
        self.assertAlmostEqual(np.exp(1), exp_scalar.val)
        self.assertAlmostEqual(-np.exp(1), exp_scalar.derv)
        exp_scalar_const = var2.exp()
        self.assertAlmostEqual(np.exp(2.2), exp_scalar_const.val)
        self.assertAlmostEqual(0, exp_scalar_const.derv)

    # test vector exponential operation
    def test_exp_vector(self):
        exp_vector = x.exp()
        self.assertAlmostEqual(np.exp(1), exp_vector.val)
        np.testing.assert_array_almost_equal(np.exp(1) * np.array([-1, 2]), exp_vector.derv)

    # test scalar inverse sine operation
    def test_scalar_arcsin(self):
        arc_sin_scalar = var9.arcsin()
        self.assertAlmostEqual(np.arcsin(0.5), arc_sin_scalar.val)
        self.assertAlmostEqual(-2 / np.sqrt((1 - 0.5 ** 2)), arc_sin_scalar.derv)

    # test vector inverse sine operation
    def test_arcsin_vector(self):
        arcsin_vector = w.arcsin()
        self.assertAlmostEqual(np.arcsin(0.5), arcsin_vector.val)
        np.testing.assert_array_almost_equal(np.array([10000, -10000]) / np.sqrt((1 - 0.5 ** 2)), arcsin_vector.derv)

    # test error handling in scalar inverse sine operation
    def test_scalar_arcsin_invalid(self):
        with self.assertRaises(ValueError) as e:
            var7.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var8.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var1.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var3.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

    # test error handling in vector inverse sine operation
    def test_scalar_arcsin_invalid(self):
        with self.assertRaises(ValueError) as e:
            x.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            y.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            t.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            p.arcsin()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

    # test scalar inverse cosine operation
    def test_scalar_arccos(self):
        arc_cos_scalar = var9.arccos()
        self.assertAlmostEqual(np.arccos(0.5), arc_cos_scalar.val)
        self.assertAlmostEqual(2 / np.sqrt((1 - 0.5 ** 2)), arc_cos_scalar.derv)

    # test vector inverse cosine operation
    def test_vector_arccos(self):
        arc_cos_vector = w.arccos()
        self.assertAlmostEqual(np.arccos(0.5), arc_cos_vector.val)
        np.testing.assert_array_almost_equal(-np.array([10000, -10000]) / np.sqrt((1 - 0.5 ** 2)), arc_cos_vector.derv)

    # test error handling in scalar inverse cosine operation
    def test_scalar_arccos_invalid(self):
        with self.assertRaises(ValueError) as e:
            var7.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var8.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var1.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var3.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

    # test error handling in vector inverse cosine operation
    def test_vector_arccos_invalid(self):
        with self.assertRaises(ValueError) as e:
            x.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            y.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            t.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            p.arccos()
        self.assertEqual("ERROR: Input to arccos() should be between -1 and 1", str(e.exception))

    # test scalar inverse tangent operation
    def test_scalar_arctan(self):
        arc_tan_scalar = var9.arctan()
        self.assertAlmostEqual(np.arctan(0.5), arc_tan_scalar.val)
        self.assertAlmostEqual(-2 / (1 + 0.5 ** 2), arc_tan_scalar.derv)

    # test vector inverse tangent operation
    def test_vector_arctan(self):
        arc_tan_vector = w.arctan()
        self.assertAlmostEqual(np.arctan(0.5), arc_tan_vector.val)
        np.testing.assert_array_almost_equal(np.array([10000, -10000]) / (1 + 0.5 ** 2), arc_tan_vector.derv)

    # test scalar power operation
    def test_scalar_power(self):
        # integer base float power
        power_scalar_int_float = var8 ** var9
        self.assertAlmostEqual(3 ** 0.5, power_scalar_int_float.val)
        self.assertAlmostEqual((3 ** 0.5) * (-2 * np.log(3) + 100000 * 0.5 / 3), power_scalar_int_float.derv)

        # integer base integer power
        power_scalar_int_int = var8 ** var8
        self.assertAlmostEqual(3 ** 3, power_scalar_int_int.val)
        self.assertAlmostEqual((3 ** 3) * (100000 * np.log(3) + 100000 * 3 / 3), power_scalar_int_int.derv)

        # float base integer power
        power_scalar_float_int = var9 ** var8
        self.assertAlmostEqual(0.5 ** 3, power_scalar_float_int.val)
        self.assertAlmostEqual((0.5 ** 3) * (100000 * np.log(0.5) + (-2) * 3 / 0.5), power_scalar_float_int.derv)

        # float base float power
        power_scalar_float_float = var9 ** var9
        self.assertAlmostEqual(0.5 ** 0.5, power_scalar_float_float.val)
        self.assertAlmostEqual((0.5 ** 0.5) * (-2 * np.log(0.5) + (-2) * 0.5 / 0.5), power_scalar_float_float.derv)

        # non-dual exponent
        power_scalar_real = var8 ** 2
        self.assertAlmostEqual(3 ** 2, power_scalar_real.val)
        self.assertAlmostEqual((3 ** 2) * (100000 * 2 / 3), power_scalar_real.derv)

        # non-dual base
        power_scalar_real = 2 ** var8
        self.assertAlmostEqual(2 ** 3, power_scalar_real.val)
        self.assertAlmostEqual((2 ** 3) * (100000 * np.log(2)), power_scalar_real.derv)

    # test vector power operation
    def test_vector_power(self):
        arc_tan_vector = x ** y
        self.assertAlmostEqual(1, arc_tan_vector.val)
        np.testing.assert_array_almost_equal(np.array([1, -2]), arc_tan_vector.derv)

    # test error handling in scalar power operation
    def test_scalar_power_invalid(self):
        with self.assertRaises(ValueError) as e:
            var6 ** var9
        self.assertEqual("ERROR: Attempted to find derivative at 0 when exponent is less than 1",
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            var7 ** var9
        self.assertEqual("ERROR: Attempted to raise a negative number to a fraction with even denominator",
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            var7 ** 0.5
        self.assertEqual("ERROR: Attempted to raise a negative number to a fraction with even denominator",
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            var6 ** 0.5
        self.assertEqual("ERROR: Attempted to find derivative at 0 when exponent is less than 1",
                         str(e.exception))

    # test error handling in vector power operation
    def test_vector_power_invalid(self):
        with self.assertRaises(ValueError) as e:
            z ** w
        self.assertEqual("ERROR: Attempted to find derivative at 0 when exponent is less than 1",
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            y ** w
        self.assertEqual("ERROR: Attempted to raise a negative number to a fraction with even denominator",
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            y ** 0.5
        self.assertEqual("ERROR: Attempted to raise a negative number to a fraction with even denominator",
                         str(e.exception))

        with self.assertRaises(ValueError) as e:
            z ** 0.5
        self.assertEqual("ERROR: Attempted to find derivative at 0 when exponent is less than 1",
                         str(e.exception))

    # test scalar logistic function
    def test_logistic_scalar(self):
        logi_scalar1 = var4.logistic()
        self.assertAlmostEqual(1 / (1 + np.exp(5.3)), logi_scalar1.val)
        self.assertAlmostEqual(124 * np.exp(5.3) / ((np.exp(5.3) + 1) ** 2), logi_scalar1.derv)
        logi_scalar2 = var5.logistic()
        self.assertAlmostEqual(1 / (1 + np.exp(-2)), logi_scalar2.val)
        self.assertAlmostEqual(6.036 * np.exp(-2) / ((np.exp(-2) + 1) ** 2), logi_scalar2.derv)

    # test vector logistic function
    def test_logistic_vector(self):
        logi_vector = x.logistic()
        self.assertAlmostEqual(1 / (1 + np.exp(-1)), logi_vector.val)
        np.testing.assert_array_almost_equal(np.array([-1, 2]) * np.exp(-1) / ((np.exp(-1) + 1) ** 2), logi_vector.derv)

    # test object string representation format
    def test_val_derv_repr(self):
        self.assertEqual("Values:1, Derivatives:-1", var1.__repr__())
        self.assertEqual("Values:-5.3, Derivatives:124", var4.__repr__())
        self.assertEqual("Values:1, Derivatives:[-1  2]", x.__repr__())

    # test value getter of object
    def test_val_derv_val(self):
        self.assertAlmostEqual(1, var1.val)
        self.assertAlmostEqual(-5.3, var4.val)
        self.assertAlmostEqual(1, x.val)

    # test derivative getter of object
    def test_val_derv_derv(self):
        self.assertAlmostEqual(-1, var1.derv)
        self.assertAlmostEqual(124, var4.derv)
        np.testing.assert_array_almost_equal([-1, 2], x.derv)

    # test derivative setter of object
    def test_val_derv_val_setter(self):
        var1.val, var4.val = -1, 10
        x.val = 2
        self.assertAlmostEqual(-1, var1.val)
        self.assertAlmostEqual(10, var4.val)
        self.assertAlmostEqual(2, x.val)

        # reset updated values back to the original setting for other tests
        var1.val = 1
        var4.val = -5.3
        x.val = 1

    # test invalid value setter of object
    def test_val_derv_val_setter_invalid(self):
        with self.assertRaises(TypeError) as e:
            var1.val = 'a'
        self.assertEqual("ERROR: Input value should be an int or float", str(e.exception))

        # reset updated values back to the original setting for other tests
        var1.val = 1

        with self.assertRaises(TypeError) as e:
            x.val = 'a'
        self.assertEqual("ERROR: Input value should be an int or float", str(e.exception))

        # reset updated values back to the original setting for other tests
        x.val = 1

    # test derivative setter of object
    def test_val_derv_derv_setter(self):
        var1.derv, var3.derv, var4.derv = 10, np.array([2]), np.array([1, 2])
        x.derv = np.array([np.pi, -np.pi])

        self.assertAlmostEqual(10, var1.derv)
        np.testing.assert_array_almost_equal([2], var3.derv)
        np.testing.assert_array_almost_equal(np.array([1, 2]), var4.derv)
        np.testing.assert_array_almost_equal(np.array([np.pi, -np.pi]), x.derv)

        # reset updated derivatives back to the original setting for other tests
        var1.derv = -1
        var3.derv = np.pi
        var4.derv = 124
        x.derv = np.array([-1, 2])

    # test invalid derivative setter of object
    def test_val_derv_derv_setter_invalid(self):
        with self.assertRaises(ValueError) as e:
            var1.derv = np.array(['a'])
        self.assertEqual("ERROR: Input value should be an int or float", str(e.exception))

        # reset updated values back to the original setting for other tests
        var1.derv = -1

        with self.assertRaises(ValueError) as e:
            x.derv = np.array(['a'])
        self.assertEqual("ERROR: Input value should be an int or float", str(e.exception))

        # reset updated values back to the original setting for other tests
        x.derv = np.array([-1, 2])

        with self.assertRaises(TypeError) as e:
            var1.derv = 'a'
        self.assertEqual("ERROR: Input value must contain an array of ints/floats or be a scalar int/float",
                         str(e.exception))

        # reset updated values back to the original setting for other tests
        var1.derv = -1

        with self.assertRaises(TypeError) as e:
            x.derv = 'a'
        self.assertEqual("ERROR: Input value must contain an array of ints/floats or be a scalar int/float",
                         str(e.exception))

        # reset updated values back to the original setting for other tests
        x.derv = np.array([-1, 2])

        with self.assertRaises(TypeError) as e:
            var1.derv = [1, 'a', 0.1]
        self.assertEqual("ERROR: Input value must contain an array of ints/floats or be a scalar int/float",
                         str(e.exception))

        # reset updated values back to the original setting for other tests
        var1.derv = -1

        with self.assertRaises(TypeError) as e:
            x.derv = [1, 'a', 0.1]
        self.assertEqual("ERROR: Input value must contain an array of ints/floats or be a scalar int/float",
                         str(e.exception))

        # reset updated values back to the original setting for other tests
        x.derv = np.array([-1, 2])

    # test type-checker of list/array contents helper function
    def test_var_type(self):
        x_1, x_2, x_3, x_4, x_5 = 1, 10, 'a', 'deriv', ['d', 'e', 'r', 'i', 'v']
        x_6, x_7, x_8, x_9, x_10 = [1, 'e', 2, 'i', 'v'], [1, 2, 3, 4, 5], \
                                   ["test", 3, 0, 'c'], [-1, 0, 10, 'a'], [0.1, 0.2, -10]

        self.assertEqual(var_type(x_1), True)
        self.assertEqual(var_type(x_2), True)
        self.assertEqual(var_type(x_3), False)
        self.assertEqual(var_type(x_4), False)
        self.assertEqual(var_type(x_5), False)
        self.assertEqual(var_type(x_6), False)
        self.assertEqual(var_type(x_7), True)
        self.assertEqual(var_type(x_8), False)
        self.assertEqual(var_type(x_9), False)
        self.assertEqual(var_type(x_10), True)

    # test equal to operator scalar
    def test_equal_scalar(self):
        var_temp = val_derv(1, 10)
        self.assertEqual(var6 == var10, (True, True))
        self.assertEqual(var6 == var11, (False, True))
        self.assertEqual(var8 == var_temp, (False, False))
        self.assertEqual(var1 == var_temp, (True, False))

    # test equal to operator vector
    def test_equal_vector(self):
        var_temp1 = val_derv(1, np.array([-1, 2]))
        var_temp2 = val_derv(1, np.array([1, -2]))
        self.assertEqual(x == var_temp1, (True, True))
        self.assertEqual(z == t, (False, True))
        self.assertEqual(x == y, (False, False))
        self.assertEqual(x == var_temp2, (True, False))

    # test not equal to operator scalar
    def test_not_equal_scalar(self):
        var_temp = val_derv(1, 10)
        self.assertEqual(var6 != var10, (False, False))
        self.assertEqual(var6 != var11, (True, False))
        self.assertEqual(var8 != var_temp, (True, True))
        self.assertEqual(var1 != var_temp, (False, True))

    # test not equal to operator vector
    def test_not_equal_vector(self):
        var_temp1 = val_derv(1, np.array([-1, 2]))
        var_temp2 = val_derv(1, np.array([1, -2]))
        self.assertEqual(x != var_temp1, (False, False))
        self.assertEqual(z != t, (True, False))
        self.assertEqual(x != y, (True, True))
        self.assertEqual(x != var_temp2, (False, True))
