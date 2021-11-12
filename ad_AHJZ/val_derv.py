# Authors: Hari Raval, Zongjun Liu Aditi Memani, Joseph Zuccarelli              #
# Course: AC 207                                                                #
# File: Regression.py                                                           #
# Description: This class defines a variable object to be used in automatic     #
# differentiation, which encapsulates the cases of both real and dual numbers   #
#################################################################################

import numpy as np


def var_type(x):
    """
    ########################
    var_type(x)

    Parameters
    ----------
    x: An object that we wish to check if the values contained within it are either integers or floats

    Returns
    -------
    True or False depending on whether all of the elements within object x are either integers or floats

    Examples
    --------
    """
    if not isinstance(x, str) and np.isscalar(x):
        return True
    else:
        for i in x:
            if isinstance(i, (float, int, np.int32, np.int64, np.float64)):
                return True
            else:
                return False


class val_derv:

    def __init__(self, val, derv_seed):
        """
        ########################
        __init__(self,value,derv_seed)

        Constructor for the val_derv class.

        Parameters
        ----------
        value: An integer or float object that represents the value of the variable
        derv_seed: An integer or float object that represents the seed value for the variable derivative

        Returns
        -------
        None
        """
        self.val = val
        self.derv = derv_seed

    def __repr__(self):
        """
        ########################
        __repr__(self)

        Operator overloading for val_derv object string representations

        Parameters
        ----------
        None

        Returns
        -------
        Formatted string representation of val_derv object
        """
        return f'Values:{self.val}, Derviatives:{self.derv}'

    @property
    def val(self):
        """
        ########################
        val(self)

        Gets the val attribute of val_derv object

        Parameters
        ----------
        None

        Returns
        -------
        val attribute of val_derv object

        Examples
        --------
        """
        return self._val

    @property
    def derv(self):
        """
        ########################
        derv(self)

        Gets the derv attribute of val_derv object

        Parameters
        ----------
        None

        Returns
        -------
        derv attribute of val_derv object

        Examples
        --------
        """
        return self._derv

    @val.setter
    def val(self, val):
        """
        ########################
        val(self, val)

        Sets the val attribute of val_derv object

        Parameters
        ----------
        val: A float or integer object that represents the value of the val_derv object

        Returns
        -------
        None

        Examples
        --------
       """
        if not var_type(val):
            raise TypeError('ERROR: Input value should be an int or float.')
        else:
            self._val = val

    @derv.setter
    def derv(self, derv):
        """
        ########################
        derv(self, derv)

        Sets the derv attribute of val_derv object

        Parameters
        ----------
        derv: A float/integer object or 1D array of float/integer objects that represents the derivative of the val_derv object

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If input contains non-integer or non-float value
            If input contains a 1D numpy array of non-integer or non-float values

        Examples
        --------
        """
        if var_type(derv):
            self._derv = derv
        elif isinstance(derv, np.ndarray) and len(derv.shape) == 1:
            try:
                derv = derv.astype(float)
            except ValueError:
                raise TypeError('ERROR: Input contains non int/float values')
            self._derv = derv
        else:
            raise TypeError('ERROR: Input contains should be an int, float, or a 1D numpy array of ints/floats.')

    def __add__(self, other):
        """
        ########################
        __add__(self, other)

        Compute the value and derivative of the addition operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the addition operation

        Examples
        --------
        """
        try:
            f = self.val + other.val
            f_prime = self.derv + other.derv
        except AttributeError:
            f = self.val + other
            f_prime = self.derv
        return val_derv(f, f_prime)

    def __sub__(self, other):
        try:
            f = self.val - other.val
            f_prime = self.derv - other.derv
        except AttributeError:
            f = self.val - other
            f_prime = self.derv
        return val_derv(f, f_prime)

    def __mul__(self, other):
        """
        ########################
        __mul__(self, other)

        Compute the value and derivative of the multiplication operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the multiplication operation

        Examples
        --------
        """
        try:
            f = self.val * other.val
            f_prime = self.val * other.derv + self.derv * other.val
        except AttributeError:
            f = self.val * other
            f_prime = self.derv * other
        return val_derv(f, f_prime)

    def __truediv__(self, other):
        """
        ########################
        __truediv__(self, other)

        Compute the value and derivative of the division operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the division operation

        Raises
        ------
        ZeroDivisionError
            If denominator in division is zero

        Examples
        --------
        """
        try:
            if other.val == 0:
                raise ZeroDivisionError("ERROR: Denominator in division should not be 0")
            f = self.val / other.val
            f_prime = (self.derv * other.val - self.val * other.derv) / other.val ** 2
            return val_derv(f, f_prime)
        except AttributeError:
            if other == 0:
                raise ZeroDivisionError("ERROR: Denominator in division should not be 0")
            f = self.val / other
            f_prime = self.derv / other
            return val_derv(f, f_prime)

    def __neg__(self):
        """
        ########################
        __neg__(self)

        Compute the value and derivative of the negation operation

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the negation operation

        Examples
        --------
        """
        return val_derv(-1 * self.val, -1 * self.derv)

    def __pow__(self, other):
        """
        ########################
        __pow__(self, power)

        Compute the value and derivative of the power operation

        Parameters
        ----------
        power: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the power operation

        Raises
        ------
        ValueError
            If negative number is raised to a fraction with an even denominator
            If exponent is less than 1 and differentiation occurs at 0

        Examples
        --------
        """
        try:
            # we did the same condition check for the dual number
            if self.val < 0 and other.val % 1 != 0 and other.val.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("ERROR: Cannot raise a negative number to a fraction with even denominator")
            if self.val == 0 and other.val < 1:
                raise ValueError("ERROR: Power function does not have a derivative at 0 if the exponent is less than 1")
            f = self.val ** other.val
            f_prime = (self.val ** (other.val - 1)) * self.derv * other.val + (
                        self.val ** other.val) * other.derv * np.log(self.val)
            return val_derv(f, f_prime)

        except AttributeError:
            if self.val < 0 and other % 1 != 0 and other.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("ERROR: Cannot raise a negative number to a fraction with even denominator")
            if self.val == 0 and other < 1:
                raise ValueError("ERROR: Power function does not have a derivative at 0 if the exponent is less than 1")
            f = self.val ** other
            f_prime = other * self.val ** (other - 1)
            return val_derv(f, self.derv * f_prime)

    def __radd__(self, other):
        """
        ########################
        __radd__(self, other)

        Compute the value and derivative of the addition operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the addition operation

        Examples
        --------
        """
        return self + other

    def __rsub__(self, other):
        """
        ########################
        __rsub__(self, other)

        Compute the value and derivative of the subtraction operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the subtraction operation

        Examples
        --------
        """
        return other + (-self)

    def __rmul__(self, other):
        """
        ########################
        __rmul__(self, other)

        Compute the value and derivative of the multiplication operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the multiplication operation

        Examples
        --------
        """
        return self * other

    def __rtruediv__(self, other):
        """
        ########################
        __rtruediv__(self, other)

        Compute the value and derivative of the division operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the division operation

        Raises
        ------
        ZeroDivisionError
            If denominator in division is zero

        Examples
        --------
        """
        try:
            if self.val == 0:
                raise ZeroDivisionError("ERROR: Denominator in division should not be 0")
            f = other.val / self.val
            f_prime = (other.derv * self.val - other.val * self.derv) / self.val ** 2
            return val_derv(f, f_prime)
        except AttributeError:
            if self.val == 0: #could be self.derv or even self double check
                raise ZeroDivisionError("ERROR: Denominator in division should not be 0")
            f = other / self.val
            f_prime = -other * self.derv / self.val ** 2
            return val_derv(f, f_prime)

    def __rpow__(self, other):
        """
        ########################
        __rpow__(self, power)

        Compute the value and derivative of the power operation

        Parameters
        ----------
        power: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the power operation

        Raises
        ------
        ValueError
            If negative number is raised to a fraction with an even denominator
            If exponent is less than 1 and differentiation occurs at 0

        Examples
        ------
        """
        return other ** self

    def sqrt(self):
        """
        ########################
        sqrt(self)

        Compute the value and derivative of the square root function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the square root function

        Examples
        --------
        """
        f = self.val ** 0.5
        f_prime = 0.5 * self.val ** (0.5 - 1)
        return val_derv(f, self.derv * f_prime)

    def log(self, base=None):
        """
        ########################
        log(self, base=None)

        Compute the value and derivative of logarthmic function (Default logarithmic base is None)

        Parameters
        ----------
        base: A float/integer object that represents the base of the logarithm (Default logarithmic base is None)

        Returns
        -------
        A val_derv object that contains the value and derivative of the logarithmic function

        Raises
        ------
        ValueError
            If self.val is less than or equal to zero
            If input base is less than or equal to zero
            If input base is equal to one

        Examples
        --------
        """
        if self.val <= 0:
            raise ValueError("ERROR: Value for log should be greater than 0")

        if base is None:
            f = np.log(self.val)
            f_prime = 1 / self.val
            return val_derv(f, self.derv * f_prime)
        else:
            if base <= 0 or base == 1:
                raise ValueError("ERROR: LOG base should be greater than 0 and not equal to 1")
            f = np.log(self.val) / np.log(base)
            f_prime_deno = self.val * np.log(base)
            f_prime = 1 / f_prime_deno
            return val_derv(f, self.derv * f_prime)

    def exp(self):
        """
        ########################
        exp(self)

        Compute the value and derivative of exponential function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the exponential function

        Examples
        --------
        """
        f = np.exp(self.val)
        f_prime = np.exp(self.val)
        return val_derv(f, self.derv * f_prime)

    def sin(self):
        """
        ########################
        sin(self)

        Compute the value and derivative of the sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the sine function

        Examples
        --------
        """
        f = np.sin(self.val)
        f_prime = np.cos(self.val)
        return val_derv(f, self.derv * f_prime)

    def cos(self):
        """
        ########################
        cos(self)

        Compute the value and derivative of the cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the cosine function

        Examples
        --------
        """
        f = np.cos(self.val)
        f_prime = - np.sin(self.val)
        return val_derv(f, self.derv * f_prime)

    def tan(self):
        """
        ########################
        tan(self)

        Compute the value and derivative of the tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the tangent function

        Raises
        ------
        ValueError
            If input is an odd multiple of pi/2

        Examples
        --------
        """
        if (self.val / (np.pi / 2)) % 2 == 1:
            raise ValueError("ERROR: Input to tan should not be an odd mutiple of pi/2")

        f = np.tan(self.val)
        f_prime = 1 / np.cos(self.val) ** 2
        return val_derv(f, self.derv * f_prime)

    def sinh(self):
        """
        ########################
        sinh(self)

        Compute the value and derivative of the hyperbolic sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic sine function

        Examples
        --------
        """
        f = np.sinh(self.val)
        f_prime = np.cosh(self.val)
        return val_derv(f, self.derv * f_prime)

    def cosh(self):
        """
        ########################
        cosh(self)

        Compute the value and derivative of the hyperbolic cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic cosine function

        Examples
        --------
        """
        f = np.cosh(self.val)
        f_prime = np.sinh(self.val)
        return val_derv(f, self.derv * f_prime)

    def tanh(self):
        """
        ########################
        tanh(self)

        Compute the value and derivative of the hyperbolic tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic tangent function

        Examples
        --------
        """
        f = np.tanh(self.val)
        f_prime = 1 / (np.cosh(self.val) ** 2)
        return val_derv(f, self.derv * f_prime)

    def arcsin(self):
        """
        ########################
        arcsin(self)

        Compute the value and derivative of the inverse sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse sine function

        Raises
        ------
        ValueError
            If input is not contained within the interval [-1,1]

        Examples
        --------
        """
        if -1 >= self.val  or self.val >= 1:
            raise ValueError("ERROR: Input to arcsin() should be between -1 and 1")
        f = np.arcsin(self.val)
        f_prime = 1 / (1 - self.val ** 2) ** 0.5
        return val_derv(f, self.derv * f_prime)

    def arctan(self):
        """
        ########################
        arctan(self)

        Compute the value and derivative of the inverse tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse tangent function

        Examples
        --------
        """
        f = np.arctan(self.val)
        f_prime = 1 / (1 + self.val ** 2)
        return val_derv(f, self.derv * f_prime)

    def arccos(self):
        """
        ########################
        arccos(self)

        Compute the value and derivative of the inverse cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse cosine function

        Raises
        ------
        ValueError
            If input is not contained within the interval [-1,1]

        Examples
        --------
        """
        if -1 >= self.val or self.val >= 1:
            raise ValueError("ERROR: Input to arccos() should be between -1 and 1")
        f = np.arccos(self.val)
        f_prime = - 1 / (1 - self.val ** 2) ** 0.5
        return val_derv(f, self.derv * f_prime)


# TODO: (in val_dervs) ADD examples to all docstrings of usage/expected with input and outputs

# TODO: (in val_dervs) in pow, we skip the check of raising negative number to a power of non-integer

# TODO: (in val_dervs) in pow, we skip power does not have a derivative at 0 if the exponent is less than 1

# TODO: (in val_dervs) rename val_derv object to something like Automatic_Diff object?

# TODO: (in val_dervs) change the set-up of the compute-derivs function

# TODO: (in forward_mode) add doc-strings for all functions in this file

# TODO: (in forward_mode) enable a user to input functions in the form 3 * x[0] + x[1] as well as 3x + y (not now)

# TODO: (in val_dervs) write test cases for the setter/getter methods in val_der.py

# TODO: (in val_dervs) write test cases for functions in forward_mode.py

# TODO: (in val_dervs) power() function: check the condition check for the dual number case and if its correct...
