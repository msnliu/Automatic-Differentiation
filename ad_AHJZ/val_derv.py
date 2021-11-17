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
    >>>x = 3
    >>>print(var_type(x))
    True
    >>>x = 'a'
    >>>print(var_type(x))
    False
    """
    if not isinstance(x, str) and np.isscalar(x):
        return True
    else:
        for i in x:
            if isinstance(i, (float, int, np.int32, np.int64, np.float64)):
                pass
            else:
                return False
        return True


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

        Examples
        --------
        >>>print(val_derv(1,2))
        Values:1, Derivatives:2
        """
        return f'Values:{self.val}, Derivatives:{self.derv}'

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
        >>>x = val_derv(1,1)
        >>>print(x.val)
        1
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
        >>>x = val_derv(1,1)
        >>>print(x.val)
        1
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
        >>>x = val_derv(1,1)
        >>>x.val = 2
        2
       """

        if var_type(val):
            self._val = val
        else:
            raise TypeError('ERROR: Input value should be an int or float.')

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
        >>>x = val_derv(1,1)
        >>>x.derv = 2
        2
        """
        if var_type(derv):
            self._derv = derv
        elif isinstance(derv, np.ndarray) and len(derv.shape) == 1:
            try:
                derv = derv.astype(float)
            except ValueError:
                raise ValueError('ERROR: Input value should be an int or float.')
            self._derv = derv
        else:
            raise TypeError('ERROR: Input value must contain an array of ints/floats or be a scalar int/float.')

    def __add__(self, other):
        """
        Compute the value and derivative of the addition operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the addition operation

        Examples
        --------
        # add of variable with scalar derivative
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 + x_2)
        Values:2, Derivatives:2

        # add of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x + y)
        Values:2, Derivatives:[1 1]
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
        """
        Compute the value and derivative of the subtraction operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the subtraction operation

        Examples
        --------
        # subtraction of variable with scalar derivative
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 - x_2)
        Values:0, Derivatives:0

        # subtraction of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x - y)
        Values:0, Derivatives:[ 1 -1]
        --------
        """

        try:
            f = self.val - other.val
            f_prime = self.derv - other.derv
        except AttributeError:
            f = self.val - other
            f_prime = self.derv
        return val_derv(f, f_prime)

    def __mul__(self, other):
        """
        Compute the value and derivative of the multiplication operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the multiplication operation

        Examples
        --------
        # multiplication of variable with scalar derivative
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 * x_2)
        Values:1, Derivatives:2

        # multiplication of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x * y)
        Values:1, Derivatives:[1 1]
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
        Compute the value and derivative of the division operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the division operation

        Raises
        ------
        ZeroDivisionError if denominator in division is zero

        Examples
        --------
        # true division of variable with scalar derivative
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 / x_2)
        Values:1.0, Derivatives:0.0

        # true division of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x / y)
        Values:1.0, Derivatives:[ 1. -1.]

        # ZeroDivisionError if denominator in division is zero
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = val_derv(0, 1)
        >>> print(x_1 / x_2)
        ZeroDivisionError: ERROR: Denominator in division should not be 0

        # ZeroDivisionError if denominator in division is zero
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = 0
        >>> print(x_1 / x_2)
        ZeroDivisionError: ERROR: Denominator in division should not be 0
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
        Compute the value and derivative of the negation operation

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the negation operation

        Examples
        --------
        # negation of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(-x)
        Values:-1, Derivatives:-1

        # negation of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(-x)
        Values:-1, Derivatives:[-1  0]
        --------
        """

        return val_derv(-1 * self.val, -1 * self.derv)

    def __pow__(self, other):
        """
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
        # power of variable with scalar derivative
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 ** x_2)
        Values:1, Derivatives:1.0

        # power of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x ** y)
        Values:1, Derivatives:[1. 0.]

        # ValueError if negative number is raised to a fraction with an even denominator
        >>> x_1 = val_derv(-1, 1)
        >>> x_2 = val_derv(0.5, 1)
        >>> print(x_1 ** x_2)
        ValueError: ERROR: Cannot raise a negative number to a fraction with even denominator

        # ValueError if exponent is less than 1 and differentiation occurs at 0
        >>> x_1 = val_derv(0, 1)
        >>> x_2 = val_derv(0.5, 1)
        >>> print(x_1 ** x_2)
        ValueError: ERROR: Power function does not have a derivative at 0 if the exponent is less than 1
        --------
        """

        try:
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
        Compute the value and derivative of the addition operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the addition operation

        Examples
        --------
        # radd of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 + x_2)
        Values:2, Derivatives:1

        # radd of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x + y)
        Values:2, Derivatives:[0 1]
        --------
        """

        return self + other

    def __rsub__(self, other):
        """
        Compute the value and derivative of the subtraction operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the subtraction operation

        Examples
        --------
        # rsubtraction of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 - x_2)
        Values:0, Derivatives:-1

        # rsubtraction of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x - y)
        Values:0, Derivatives:[ 0 -1]
        --------
        """

        return other + (-self)

    def __rmul__(self, other):
        """
        Compute the value and derivative of the multiplication operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the multiplication operation

        Examples
        --------
        # rmul of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 * x_2)
        Values:1, Derivatives:1

        # rmul of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x * y)
        Values:1, Derivatives:[0 1]
        --------
        """

        return self * other

    def __rtruediv__(self, other):
        """
        Compute the value and derivative of the division operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the division operation

        Raises
        ------
        ZeroDivisionError if denominator in division is zero

        Examples
        --------
        # reverse true division of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 / x_2)
        Values:1.0, Derivatives:-1.0

        # reverse true division of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x / y)
        Values:1.0, Derivatives:[ 0. -1.]

        # ZeroDivisionError if denominator in division is zero
        >>> x_1 = 1
        >>> x_2 = val_derv(0, 1)
        >>> print(x_1 / x_2)
        ZeroDivisionError: ERROR: Denominator in division should not be 0
        --------
        """

        if self.val == 0:
            raise ZeroDivisionError("ERROR: Denominator in division should not be 0")
        f = other / self.val
        f_prime = - other * self.derv / self.val ** 2
        return val_derv(f, f_prime)

    def __rpow__(self, other):
        """
        Compute the value and derivative of the power operation

        Parameters
        ----------
        power: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the power operation

        Examples
        --------
        # reverse power of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 ** x_2)
        Values:1, Derivatives:0.0

        # reverse power of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x ** y)
        Values:1, Derivatives:[0. 0.]
        ------
        """

        f = other ** self.val
        f_prime = (other ** self.val) * self.derv * np.log(other)
        return val_derv(f, f_prime)

    def sqrt(self):
        """
        Compute the value and derivative of the square root function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the square root function

        Examples
        --------
        # sqrt of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(x.sqrt())
        Values:1.0, Derivatives:0.5

        # sqrt of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(x.sqrt())
        Values:1.0, Derivatives:[0.5 0. ]
        --------
        """

        return self.__pow__(0.5)

    def log(self, base=None):
        """
        Compute the value and derivative of logarthmic function (Default logarithmic base is None)
        Parameters

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
        # ValueError if self.val is less than ot equal to zero
        >>> x = val_derv(0, 1)
        >>> print(x.log())
        ValueError: ERROR: Value for log should be greater than 0

        >>> x = val_derv(-1, 1)
        >>> print(x.log())
        ValueError: ERROR: Value for log should be greater than 0

        # ValueError if input base is less than or equal to zero
        >>> x = val_derv(1, 1)
        >>> print(x.log(base = -1))
        ValueError: ERROR: LOG base should be greater than 0 and not equal to 1

        >>> x = val_derv(1, 1)
        >>> print(x.log(base = 1))
        ValueError: ERROR: LOG base should be greater than 0 and not equal to 1

        >>> x = val_derv(1, 1)
        >>> print(x.log(base = 0))
        ValueError: ERROR: LOG base should be greater than 0 and not equal to 1

        # log of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(x.log())
        Values:0.0, Derivatives:1.0

        # log of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(x.log())
        Values:0.0, Derivatives:[1. 0.]
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
        Compute the value and derivative of exponential function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the exponential function

        Examples
        --------
        # exp of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.exp())
        Values:1.0, Derivatives:1.0

        # exp of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.exp())
        Values:1.0, Derivatives:[1. 0.]
        --------
        """

        f = np.exp(self.val)
        f_prime = np.exp(self.val)
        return val_derv(f, self.derv * f_prime)

    def sin(self):
        """
        Compute the value and derivative of the sine function
        Parameters

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the sine function

        Examples
        --------
        # sin of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.sin())
        Values:0.0, Derivatives:1.0

        # sin of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.sin())
        Values:0.0, Derivatives:[1. 0.]
        --------
        """

        f = np.sin(self.val)
        f_prime = np.cos(self.val)
        return val_derv(f, self.derv * f_prime)

    def cos(self):
        """
        Compute the value and derivative of the cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the cosine function

        Examples
        --------
        # cos of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.cos())
        Values:1.0, Derivatives:-0.0

        # cos of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.cos())
        Values:1.0, Derivatives:[-0. -0.]
        --------
        """
        f = np.cos(self.val)
        f_prime = - np.sin(self.val)
        return val_derv(f, self.derv * f_prime)

    def tan(self):
        """
        Compute the value and derivative of the tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the tangent function

        Raises
        ------
        ValueError if input is an odd multiple of pi/2

        Examples
        --------
        # tan of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.tan())
        Values:0.0, Derivatives:1.0

        # tan of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.tan())
        Values:0.0, Derivatives:[1. 0.]

        # ValueError if input is an odd multiple of pi/2
        >>> x = val_derv(np.pi / 2, 1)
        >>> print(x.tan())
        ValueError: ERROR: Input to tan should not be an odd mutiple of pi/2
        --------
        """

        if (self.val / (np.pi / 2)) % 2 == 1:
            raise ValueError("ERROR: Input to tan should not be an odd mutiple of pi/2")

        f = np.tan(self.val)
        f_prime = 1 / np.cos(self.val) ** 2
        return val_derv(f, self.derv * f_prime)

    def sinh(self):
        """
        Compute the value and derivative of the hyperbolic sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic sine function

        Examples
        --------
        # sinh of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.sinh())
        Values:0.0, Derivatives:1.0

        # sinh of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.sinh())
        Values:0.0, Derivatives:[1. 0.]
        --------
        """

        f = np.sinh(self.val)
        f_prime = np.cosh(self.val)
        return val_derv(f, self.derv * f_prime)

    def cosh(self):
        """
        Compute the value and derivative of the hyperbolic cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic cosine function

        Examples
        --------
        # cosh of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.cosh())
        Values:1.0, Derivatives:0.0

        # cosh of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.cosh())
        Values:1.0, Derivatives:[0. 0.]
        --------
        """

        f = np.cosh(self.val)
        f_prime = np.sinh(self.val)
        return val_derv(f, self.derv * f_prime)

    def tanh(self):
        """
        Compute the value and derivative of the hyperbolic tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic tangent function

        Examples
        --------
        # tanh of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.tanh())
        Values:0.0, Derivatives:1.0

        # tanh of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.tanh())
        Values:0.0, Derivatives:[1. 0.]
        --------
        """

        f = np.tanh(self.val)
        f_prime = 1 / (np.cosh(self.val) ** 2)
        return val_derv(f, self.derv * f_prime)

    def arcsin(self):
        """
        Compute the value and derivative of the inverse sine function

        Parameters
        ----------
        None
        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse sine function

        Raises
        ------
        ValueError if input is not contained within the interval [-1,1]

        Examples
        --------
        # arcsin of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.arcsin())
        Values:0.0, Derivatives:1.0

        # arcsin of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.arcsin())
        Values:0.0, Derivatives:[1. 0.]

        # ValueError for input outside of the interval -1 to 1
        >>> x = val_derv(2, 1)
        >>> print(x.arcsin())
        ValueError: ERROR: Input to arcsin() should be between -1 and 1
        --------
        """

        if -1 >= self.val or self.val >= 1:
            raise ValueError("ERROR: Input to arcsin() should be between -1 and 1")
        f = np.arcsin(self.val)
        f_prime = 1 / (1 - self.val ** 2) ** 0.5
        return val_derv(f, self.derv * f_prime)

    def arctan(self):
        """
        Compute the value and derivative of the inverse tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse tangent function

        Examples
        --------
        # arctan of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(x.arctan())
        Values:0.7853981633974483, Derivatives:0.5

        # arctan of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(x.arctan())
        Values:0.7853981633974483, Derivatives:[0.5 0. ]
        """

        f = np.arctan(self.val)
        f_prime = 1 / (1 + self.val ** 2)
        return val_derv(f, self.derv * f_prime)

    def arccos(self):
        """
        Compute the value and derivative of the inverse cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse cosine function

        Raises
        ------
        ValueError will be raised if input is not contained within the interval [-1,1]

        Examples
        --------
        # arccos of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.arccos())
        Values:1.5707963267948966, Derivatives:-1.0

        # arccos of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.arccos())
        Values:1.5707963267948966, Derivatives:[-1. -0.]

        # ValueError for input outside of the interval -1 to 1
        >>> x = val_derv(2, 1)
        >>> print(x.arccos())
        ValueError: ERROR: Input to arccos() should be between -1 and 1
        """

        if -1 >= self.val or self.val >= 1:
            raise ValueError("ERROR: Input to arccos() should be between -1 and 1")
        f = np.arccos(self.val)
        f_prime = - 1 / (1 - self.val ** 2) ** 0.5
        return val_derv(f, self.derv * f_prime)

