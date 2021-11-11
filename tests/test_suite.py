#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:11:04 2021

@author: aditimemani
"""
import unittest
import numpy as np

from val_derv import val_derv

var1 = val_derv(1, 1)
var2 = val_derv(2.2, 1)
var3 = val_derv(-1, 1)
var4 = val_derv(-5.3, 1)
var5 = val_derv(2, 1)
var6 = val_derv(0, 1)
var7 = val_derv(-3, 1)
var8 = val_derv(3, 1)
var9 = val_derv(0.5, 1)


class Val_Derv_Elem_Test(unittest.TestCase):

    def test_scalar_add(self):
        sum1_val = var1.val + var2.val
        sum1_derv = var1.derv + var2.derv
        sum1_rev_val = var2.val + var1.val
        sum1_rev_derv = var2.derv + var1.derv
        sum2_val = var1.val + 2
        sum2_rev_val = 2 + var1.val
        sum3_val = var1 + 2
        sum3_rev_val = 2 + var1
        self.assertAlmostEqual(3.2, sum1_val)
        self.assertAlmostEqual(2, sum1_derv)
        self.assertAlmostEqual(3.2, sum1_rev_val)
        self.assertAlmostEqual(2, sum1_rev_derv)
        self.assertAlmostEqual(3, sum2_val)
        self.assertAlmostEqual(3, sum2_rev_val)
        self.assertAlmostEqual(3, sum3_val)
        self.assertAlmostEqual(3, sum3_rev_val)

    def test_scalar_sub(self):
        sub1_val = var1.val - var2.val
        sub1_derv = var1.derv - var2.derv
        sub1_rev_val = var2.val - var1.val
        sub1_rev_derv = var2.derv - var1.derv
        sub2_val = var1.val - 2
        sub2_rev_val = 2 - var1.val
        sub3_val = var1 - 2
        sub3_rev_val = 2 - var1
        self.assertAlmostEqual(-1.2, sub1_val)
        self.assertAlmostEqual(0, sub1_derv)
        self.assertAlmostEqual(1.2, sub1_rev_val)
        self.assertAlmostEqual(0, sub1_rev_derv)
        self.assertAlmostEqual(1.2, sub2_val)
        self.assertAlmostEqual(0, sub2_rev_val)
        self.assertAlmostEqual(-1, sub3_val)
        self.assertAlmostEqual(1, sub3_rev_val)

    def test_scalar_mul(self):
        prod1_val = var1.val * var2.val
        prod1_derv = var1.derv * var2.derv
        prod1_rev_val = var2.val * var1.val
        prod1_rev_derv = var2.derv * var1.derv
        prod2_val = var1.val * 2
        prod2_rev_val = 2 * var1.val
        prod3_val = var1 * 2
        prod3_rev_val = 2 * var1
        self.assertAlmostEqual(2.2, prod1_val)
        self.assertAlmostEqual(1, prod1_derv)
        self.assertAlmostEqual(2.2, prod1_rev_val)
        self.assertAlmostEqual(1, prod1_rev_derv)
        self.assertAlmostEqual(2, prod2_val)
        self.assertAlmostEqual(2, prod2_rev_val)
        self.assertAlmostEqual(2, prod3_val)
        self.assertAlmostEqual(2, prod3_rev_val)

    def test_scalar_trudiv(self):
        div1_val = var1.val / var5.val
        div1_derv = var1.derv / var5.derv
        div1_rev_val = var5.val / var1.val
        div1_rev_derv = var5.derv / var1.derv
        div2_val = var1.val / 2
        div2_rev_val = 2 / var1.val
        div3_val = var1 / 2
        div3_rev_val = 2 / var1
        self.assertAlmostEqual(0.5, div1_val)
        self.assertAlmostEqual(1, div1_derv)
        self.assertAlmostEqual(2, div1_rev_val)
        self.assertAlmostEqual(1, div1_rev_derv)
        self.assertAlmostEqual(0.5, div2_val)
        self.assertAlmostEqual(2, div2_rev_val)
        self.assertAlmostEqual(0.5, div3_val)
        self.assertAlmostEqual(2, div3_rev_val)

    def test_scalar_truedive_zeroError(self):
        with self.assertRaises(ZeroDivisionError) as e:
            div1_val = var1.val / 0
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

        with self.assertRaises(ZeroDivisionError) as e:
            div2_val = var1.val / var6.val
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

        with self.assertRaises(ZeroDivisionError) as e:
            div3_val = 3 / var6.val
        self.assertEqual('ERROR: Denominator in division should not be 0', str(e.exception))

    def test_scalar_neg(self):
        div1_val = -var1.val
        div1_derv = -var1.derv
        div2_val = -var4.val
        div2_derv = -var4.derv
        self.assertAlmostEqual(-1, div1_val)
        self.assertAlmostEqual(-1, div1_derv)
        self.assertAlmostEqual(5.3, div2_val)
        self.assertAlmostEqual(-1, div2_derv)

    def test_sin_scalar(self):
        result1 = var5.sin()
        result2 = var2.sin()

        self.assertAlmostEqual(np.sin(2), result1.val)
        self.assertAlmostEqual(np.cos(2) * 1, result1.derv)
        self.assertAlmostEqual(np.sin(2.2), result2.val)
        self.assertAlmostEqual(np.cos(2.2) * 1, result2.derv)

    def test_cos_scalar(self):
        result1 = var5.cos()
        result2 = var2.cos()

        self.assertAlmostEqual(np.cos(2), result1.val)
        self.assertAlmostEqual(-np.sin(2) * 1, result1.derv)
        self.assertAlmostEqual(np.cos(2.2), result2.val)
        self.assertAlmostEqual(-np.sin(2.2) * 1, result2.derv)

    def test_tan_scalar(self):
        result1 = var5.tan()
        result2 = var2.tan()

        self.assertAlmostEqual(np.tan(2), result1.val)
        self.assertAlmostEqual(-2 / (np.cos(2) ** 2), result1.derv)
        self.assertAlmostEqual(np.tan(2.2), result2.val)
        self.assertAlmostEqual(-2 / (np.cos(2.2) ** 2), result2.deriv)

    def test_tan_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            var = val_derv(3 * np.pi / 2, 1)
            var.tan()
        self.assertEqual("ERROR: Input to tan should not be an odd mutiple of pi/2", str(e.exception))

    def test_sinh_scalar(self):
        result1 = var5.sinh()
        result2 = var2.sinh()

        self.assertAlmostEqual(np.sinh(2), result1.val)
        self.assertAlmostEqual(np.cosh(2) * 1, result1.derv)
        self.assertAlmostEqual(np.sinh(2.2), result2.val)
        self.assertAlmostEqual(np.cosh(2.2) * 1, result2.derv)

    def test_cosh_scalar(self):
        result1 = var5.cosh()
        result2 = var2.cosh()

        self.assertAlmostEqual(np.cosh(2), result1.val)
        self.assertAlmostEqual(np.sinh(2) * 1, result1.derv)
        self.assertAlmostEqual(np.cosh(2.2), result2.val)
        self.assertAlmostEqual(np.sinh(2.2) * 1, result2.derv)

    def test_tanh_scalar(self):
        result1 = var5.tanh()
        result2 = var2.tanh()

        self.assertAlmostEqual(np.tanh(2), result1.val)
        self.assertAlmostEqual((1 - np.tanh(2) ** 2) * 1, result1.derv)
        self.assertAlmostEqual(np.tanh(2.2), result2.val)
        self.assertAlmostEqual((1 - np.tanh(2.2) ** 2) * 1, result2.derv)

    def test_log_scalar(self):
        result = var2.log()
        self.assertAlmostEqual(np.log(2.2), result.val)
        self.assertAlmostEqual(1 * 1, result.derv)

        result2 = var2.log(10)
        self.assertAlmostEqual(np.log(2.2) / np.log(10), result2.val)
        self.assertAlmostEqual((2.2 * np.log(10)) * 1, result2.derv)

    def test_log_scalar_invalid_value(self):
        with self.assertRaises(ValueError) as e:
            result = var3.log()
        self.assertEqual("ERROR: Value for log should be greater than 0", str(e.exception))
        with self.assertRaises(ValueError) as e:
            result2 = var2.log(1)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))
        with self.assertRaises(ValueError) as e:
            result3 = var2.log(0)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))
        with self.assertRaises(ValueError) as e:
            result3 = var2.log(-1)
        self.assertEqual("ERROR: LOG base should be greater than 0 and not equal to 1", str(e.exception))

    def test_exp_scalar(self):
        result1 = var2.exp()
        result2 = var1.exp()

        self.assertAlmostEqual(np.exp(2.2), result1.val)
        self.assertAlmostEqual(np.exp(2.2) * 1, result1.derivative)
        self.assertAlmostEqual(np.exp(1), result2.val)
        self.assertAlmostEqual(np.exp(1) * 1, result2.derivative)

    def test_scalar_arcsin(self):
        arc_sin_res = var9.arcsin()
        self.assertAlmostEqual(np.arcsin(0.5), arc_sin_res.val)
        self.assertAlmostEqual(1 / np.sqrt((1 - 0.5 ** 2)), arc_sin_res.derv)

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

    def test_scalar_arccos(self):
        arc_cos_res = var9.arccos()
        self.assertAlmostEqual(np.arccos(0.5), arc_cos_res.val)
        self.assertAlmostEqual(- 1 / np.sqrt((1 - 0.5 ** 2)), arc_cos_res.derv)

    def test_scalar_arccos_invalid(self):
        with self.assertRaises(ValueError) as e:
            var7.arccos()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var8.arccos()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var1.arccos()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var3.arccos()
        self.assertEqual("ERROR: Input to arcsin() should be between -1 and 1", str(e.exception))

    def test_scalar_arctan(self):
        arc_tan_res_float = var9.arctan()
        self.assertAlmostEqual(np.arctan(0.5), arc_tan_res_float.val)

        self.assertAlmostEqual(1 / (1 + 0.5 ** 2), arc_tan_res_float.derv)
        arc_tan_res_int = var8.arctan()
        self.assertAlmostEqual(np.arctan(3), arc_tan_res_int.val)
        self.assertAlmostEqual(1 / (1 + 3 ** 2), arc_tan_res_int.derv)

    def test_scalar_power(self):
        # integer base float power
        power_res_int_float = var8 ** var9
        self.assertAlmostEqual(3 ** 0.5, power_res_int_float.val)
        self.assertAlmostEqual((3 ** 0.5) * (np.log(3) + 0.5 / 3), power_res_int_float.derv)
        # integer base integer power
        power_res_int_int = var8 ** var8
        self.assertAlmostEqual(3 ** 3, power_res_int_int.val)
        self.assertAlmostEqual((3 ** 3)(np.log(3) + 3 / 3), power_res_int_int.derv)
        # float base integer power
        power_res_float_int = var9 ** var8
        self.assertAlmostEqual(0.5 ** 3, power_res_float_int.val)
        self.assertAlmostEqual((0.5 ** 3) * (np.log(0.5) + 3 / 0.5), power_res_float_int.derv)
        # float base float power
        power_res_float_float = var9 ** var9
        self.assertAlmostEqual(0.5 ** 0.5, power_res_float_int.val)
        self.assertAlmostEqual((0.5 ** 0.5) * (np.log(0.5) + 0.5 / 0.5), power_res_float_float.derv)

        power_res_real = var8 ** 2
        self.assertAlmostEqual(3 ** 2, power_res_real.val)
        self.assertAlmostEqual((3 ** 2)(2 / 3), power_res_real.derv)

    def test_scalar_power_invalid(self):
        with self.assertRaises(ValueError) as e:
            var1 ** var9
        self.assertEqual("ERROR: Power function does not have a derivative at 0 if the exponent is less than 1", str(e.exception))

        with self.assertRaises(ValueError) as e:
            var7 ** var9
        self.assertEqual("ERROR: Cannot raise a negative number to a fraction with even denominator", str(e.exception))
