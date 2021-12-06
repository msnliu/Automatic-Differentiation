# Authors: Hari Raval, Zongjun Liu, Aditi Memani, Joseph Zuccarelli             #
# Course: AC 207                                                                #
# File: val_derv.py                                                             #
# Description: This class defines a variable object to be used in automatic     #
# differentiation, which encapsulates the cases of both real and dual numbers   #
#################################################################################

import numpy as np


def var_type(x):
    """
    method to check whether input x contains numeric elements or not

    Parameters
    ----------
    x: An object to be checked if the values contained within it are either integers or floats

    Returns
    -------
    True or False depending on whether all of the elements within object x are either integers or floats

    Examples
    --------
    >>> x = 3
    >>> print(var_type(x))
    True
    >>> x = 'a'
    >>> print(var_type(x))
    False

    """

    # if the input object is not a character and is a scalar, it must be numeric
    if not isinstance(x, str) and np.isscalar(x):
        return True
    # if the input object is of an "array" or "list" type, check all contents
    else:
        for i in x:
            # if current element is numeric, move on to the next
            if isinstance(i, (float, int, np.int32, np.int64, np.float64)):
                pass
            else:
                return False

        return True


class val_derv:
    """
    A class representing a custom variable object to be used in automatic differentiation.
    The class encapsulates the cases of both real and dual numbers, containing overloaded dunder methods
    and overloaded elementary functions that are used in forward mode calculation.

    Instance Variables
    ----------
    val: value of the val_derv object
    derv: derivative(s) of the val_derv object

    Returns
    -------
    A val_derv object that contains the value and derivative

    Examples
    --------
    # sample instantiation and use case of val_derv object
    >>> x_1 = val_derv(1, 1)
    >>> x_2 = val_derv(1, 1)
    >>> print(x_1 + x_2)
    Values:2, Derivatives:2

    """

    def __init__(self, val, derv_seed):
        """
        constructor to create a val_derv object represented by a value and derivative

        Parameters
        ----------
        val: An integer or float object that represents the variable value
        derv_seed: An integer or float object that represents the seed value for the variable derivative

        Returns
        -------
        None

        """

        self.val = val
        self.derv = derv_seed

    @property
    def val(self):
        """
        method to retrieve the value attribute of a val_derv object

        Parameters
        ----------
        None

        Returns
        -------
        val attribute of val_derv object

        Examples
        --------
        >>> x = val_derv(1, 1)
        >>> print(x.val)
        1

        """

        return self._val

    @val.setter
    def val(self, val):
        """
        method to set the value attribute of val_derv object

        Parameters
        ----------
        val: A float or integer object that represents the value of the val_derv object

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
        >>> x = val_derv(1, 1)
        >>> x.val = 2
        2

       """

        if var_type(val):
            self._val = val
        # ensure user does not pass in non-numeric input
        else:
            raise TypeError('ERROR: Input value should be an int or float')

    @property
    def derv(self):
        """
        method to retrieve the derivative attribute of val_derv object

        Parameters
        ----------
        None

        Returns
        -------
        derv attribute of val_derv object

        Examples
        --------
        >>> x = val_derv(1, 1)
        >>> print(x.val)
        1

        """

        return self._derv

    @derv.setter
    def derv(self, derv):
        """
        method to set the derivative attribute of val_derv object

        Parameters
        ----------
        derv: A float/integer object or 1D array of float/integer objects that represents val_derv derivative

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
        >>> x = val_derv(1, 1)
        >>> x.derv = 2
        2

        """

        # ensure the derivative is numeric before setting val_derv attribute
        if var_type(derv):
            self._derv = derv
        # in the case of a 1D array of derivatives, check each element individually
        elif isinstance(derv, np.ndarray) and len(derv.shape) == 1:
            try:
                derv = derv.astype(float)
            except ValueError:
                raise ValueError('ERROR: Input value should be an int or float')
            self._derv = derv
        # for all other non-numeric cases, raise the appropriate value error
        else:
            raise TypeError('ERROR: Input value must contain an array of ints/floats or be a scalar int/float')

    def __repr__(self):
        """
        method to overload the string representation for val_derv object

        Parameters
        ----------
        None

        Returns
        -------
        Formatted string representation of val_derv object

        Examples
        --------
        >>> print(val_derv(1, 2))
        Values:1, Derivatives:2

        """

        return f'Values:{self.val}, Derivatives:{self.derv}'

    def __add__(self, other):
        """
        method to compute the value and derivative of the addition operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

        Returns
        -------
        A val_derv object that contains the value and derivative of the addition operation

        Examples
        --------
        # add of variables with scalar derivative
        >>> x_1 = val_derv(1, 1)
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 + x_2)
        Values:2, Derivatives:2

        # add of variables with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x + y)
        Values:2, Derivatives:[1 1]

        """

        # perform addition assuming other is a dual number
        try:
            f = self.val + other.val
            f_prime = self.derv + other.derv
        # perform addition when other is a real number
        except AttributeError:
            f = self.val + other
            f_prime = self.derv
        return val_derv(f, f_prime)

    def __sub__(self, other):
        """
        method to compute the value and derivative of the subtraction operation

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

        """

        # perform subtraction assuming other is a dual number
        try:
            f = self.val - other.val
            f_prime = self.derv - other.derv
        # perform subtraction when other is a real number
        except AttributeError:
            f = self.val - other
            f_prime = self.derv
        return val_derv(f, f_prime)

    def __mul__(self, other):
        """
        method to compute the value and derivative of the multiplication operation

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

        """

        # perform multiplication assuming other is a dual number
        try:
            f = self.val * other.val
            f_prime = self.val * other.derv + self.derv * other.val
        # perform multiplication when other is a real number
        except AttributeError:
            f = self.val * other
            f_prime = self.derv * other
        return val_derv(f, f_prime)

    def __truediv__(self, other):
        """
        method to compute the value and derivative of the division operation

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

        # perform division assuming other is a dual number
        try:
            # ensure the user does not divide by 0
            if other.val == 0:
                raise ZeroDivisionError("ERROR: Denominator in division should not be 0")
            f = self.val / other.val
            f_prime = (self.derv * other.val - self.val * other.derv) / other.val ** 2
            return val_derv(f, f_prime)
        # perform division when other is a real number
        except AttributeError:
            # ensure the user does not divide by 0
            if other == 0:
                raise ZeroDivisionError("ERROR: Denominator in division should not be 0")
            f = self.val / other
            f_prime = self.derv / other
            return val_derv(f, f_prime)

    def __neg__(self):
        """
        method to compute the value and derivative of the negation operation

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

        """

        return val_derv(-1 * self.val, -1 * self.derv)

    def __pow__(self, other):
        """
        method to compute the value and derivative of the power operation

        Parameters
        ----------
        other: A float/integer object or val_derv object

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
        ValueError: ERROR: Attempted to raise a negative number to a fraction with even denominator

        # ValueError if exponent is less than 1 and differentiation occurs at 0
        >>> x_1 = val_derv(0, 1)
        >>> x_2 = val_derv(0.5, 1)
        >>> print(x_1 ** x_2)
        ValueError: ERROR: Attempted to find derivative at 0 when exponent is less than 1

        """

        # perform power operation assuming other is a dual number
        try:
            # ensure user does not raise a negative number to a fraction with an even denominator
            if self.val < 0 and other.val % 1 != 0 and other.val.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("ERROR: Attempted to raise a negative number to a fraction with even denominator")
            # ensure user does not have a 0 derivative when the exponent is less than 1
            if self.val == 0 and other.val < 1:
                raise ValueError("ERROR: Attempted to find derivative at 0 when exponent is less than 1")

            f = self.val ** other.val
            # compute the derivative power rule for a dual number exponent
            f_prime = (self.val ** (other.val - 1)) * self.derv * other.val + (
                    self.val ** other.val) * other.derv * np.log(self.val)
            return val_derv(f, f_prime)

        # perform power operation when other is a real number
        except AttributeError:
            # ensure user does not raise a negative number to a fraction with an even denominator
            if self.val < 0 and other % 1 != 0 and other.as_integer_ratio()[1] % 2 == 0:
                raise ValueError("ERROR: Attempted to raise a negative number to a fraction with even denominator")
            # ensure user does not have a 0 derivative when the exponent is less than 1
            if self.val == 0 and other < 1:
                raise ValueError("ERROR: Attempted to find derivative at 0 when exponent is less than 1")

            f = self.val ** other
            # compute the derivative power rule for a real number exponent
            f_prime = other * self.val ** (other - 1)
            return val_derv(f, self.derv * f_prime)

    def __radd__(self, other):
        """
        method to compute the value and derivative of the reverse addition operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the reverse addition operation

        Examples
        --------
        # reverse addition of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 + x_2)
        Values:2, Derivatives:1

        # reverse addition of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x + y)
        Values:2, Derivatives:[0 1]

        """

        return self + other

    def __rsub__(self, other):
        """
        method to compute the value and derivative of the reverse subtraction operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the reverse subtraction operation

        Examples
        --------
        # reverse subtraction of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 - x_2)
        Values:0, Derivatives:-1

        # reverse subtraction of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x - y)
        Values:0, Derivatives:[ 0 -1]

        """

        return other + (-self)

    def __rmul__(self, other):
        """
        method to compute the value and derivative of the reverse multiplication operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the reverse multiplication operation

        Examples
        --------
        # reverse multiplication of variable with scalar derivative
        >>> x_1 = 1
        >>> x_2 = val_derv(1, 1)
        >>> print(x_1 * x_2)
        Values:1, Derivatives:1

        # reverse multiplication of variable with vector derivative
        >>> x = 1
        >>> y = val_derv(1, np.array([0, 1]))
        >>> print(x * y)
        Values:1, Derivatives:[0 1]

        """

        return self * other

    def __rtruediv__(self, other):
        """
        method to compute the value and derivative of the reverse division operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the reverse division operation

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

        """

        # ensure the user does not attempt to divide by 0
        if self.val == 0:
            raise ZeroDivisionError("ERROR: Denominator in division should not be 0")

        # compute the value and derivative of the reverse division analogus to forward direction
        f = other / self.val
        f_prime = - other * self.derv / self.val ** 2
        return val_derv(f, f_prime)

    def __rpow__(self, other):
        """
        method to compute the value and derivative of the reverse power operation

        Parameters
        ----------
        other: A float/integer object

        Returns
        -------
        A val_derv object that contains the value and derivative of the reverse power operation

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

        """

        # compute value and derivative of the reverse power function analogous to forward direction
        f = other ** self.val
        f_prime = (other ** self.val) * self.derv * np.log(other)
        return val_derv(f, f_prime)

    def __eq__(self, other):
        """
        method to check whether the value and derivative of val_derv objects are equal

        Parameters
        ----------
        other: A val_derv object

        Returns
        -------
        A tuple of boolean entries where the first entry determines if the function values are equal and the second
        entry determines if the derivative values are equal

        Examples
        --------
        # equality of val_derv objects with the same contents
        >>> val_derv1 = val_derv(1, 10)
        >>> val_derv2 = val_derv(1, 10)
        >>> print(val_derv1 == val_derv2)
        (True, True)

        # equality of val_derv objects with partially different contents
        >>> val_derv1 = val_derv(1, 10)
        >>> val_derv2 = val_derv(2, 10)
        >>> print(val_derv1 == val_derv2)
        (False, True)

        # equality of val_derv objects with all different contents
        >>> val_derv1 = val_derv(2, 1)
        >>> val_derv2 = val_derv(1, 2)
        >>> print(val_derv1 == val_derv2)
        (False, False)

        """

        # check if val_derv values are equal
        try:
            value_eq = all(self.val == other.val)
        except TypeError:
            value_eq = True if self.val == other.val else False

        # check if val_derv derivatives are equal
        try:
            derivative_eq = all(self.derv == other.derv)
        except TypeError:
            derivative_eq = True if self.derv == other.derv else False

        return value_eq, derivative_eq

    def __ne__(self, other):
        """
        method to check whether the value and derivative of val_derv objects are not equal

        Parameters
        ----------
        other: A val_derv object

        Returns
        -------
        A tuple of boolean entries where the first entry determines if the function values are not equal and
        the second entry determines if the derivative values are not equal

        Examples
        --------
        # non-equality of val_derv objects with the same contents
        >>> val_derv1 = val_derv(1, 10)
        >>> val_derv2 = val_derv(1, 10)
        >>> print(val_derv1 != val_derv2)
        (False, False)

        # non-equality of val_derv objects with partially different contents
        >>> val_derv1 = val_derv(1, 10)
        >>> val_derv2 = val_derv(2, 10)
        >>> print(val_derv1 != val_derv2)
        (True, False)

        # non-equality of val_derv objects with all different contents
        >>> val_derv1 = val_derv(2, 1)
        >>> val_derv2 = val_derv(1, 2)
        >>> print(val_derv1 != val_derv2)
        (True, True)

        """

        # check if val_derv values are not equal
        try:
            value_eq = all(self.val != other.val)
        except TypeError:
            value_eq = True if self.val != other.val else False

        # check if val_derv derivatives are not equal
        try:
            derivative_eq = all(self.derv != other.derv)
        except TypeError:
            derivative_eq = True if self.derv != other.derv else False

        return value_eq, derivative_eq

    def sqrt(self):
        """
        method to compute the value and derivative of the square root function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the square root function

        Examples
        --------
        # square root of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(x.sqrt())
        Values:1.0, Derivatives:0.5

        # square root of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(x.sqrt())
        Values:1.0, Derivatives:[0.5 0. ]

        """

        return self.__pow__(0.5)

    def log(self, base=None):
        """
        method to compute the value and derivative of logarthmic function
        Parameters

        Parameters
        ----------
        base: A float/integer object that represents the base of the logarithm (default logarithmic base is None)

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
        # ValueError if self.val is less than or equal to zero
        >>> x = val_derv(0, 1)
        >>> print(x.log())
        ValueError: ERROR: Value for log should be greater than 0
        >>> x = val_derv(-1, 1)
        >>> print(x.log())
        ValueError: ERROR: Value for log should be greater than 0

        # ValueError if input base is less than or equal to zero or equal to 1
        >>> x = val_derv(1, 1)
        >>> print(x.log(base = -1))
        ValueError: ERROR: LOG base should be greater than 0 and not equal to 1
        >>> x = val_derv(1, 1)
        >>> print(x.log(base = 1))
        ValueError: ERROR: LOG base should be greater than 0 and not equal to 1
        >>> x = val_derv(1, 1)
        >>> print(x.log(base = 0))
        ValueError: ERROR: LOG base should be greater than 0 and not equal to 1

        # logarithm of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(x.log())
        Values:0.0, Derivatives:1.0

        # logarithm of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(x.log())
        Values:0.0, Derivatives:[1. 0.]

        """

        # ensure user does not pass a negative value to take the log of
        if self.val <= 0:
            raise ValueError("ERROR: Value for log should be greater than 0")

        # if the default base is used, proceed without checking base conditions
        if base is None:
            f = np.log(self.val)
            f_prime = 1 / self.val
            return val_derv(f, self.derv * f_prime)
        # ensure the user specifies a valid base before computing the log value and derivative
        else:
            if base <= 0 or base == 1:
                raise ValueError("ERROR: LOG base should be greater than 0 and not equal to 1")
            f = np.log(self.val) / np.log(base)
            f_prime_deno = self.val * np.log(base)
            f_prime = 1 / f_prime_deno
            return val_derv(f, self.derv * f_prime)

    def exp(self):
        """
        method to compute the value and derivative of exponential function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the exponential function

        Examples
        --------
        # exponential of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.exp())
        Values:1.0, Derivatives:1.0

        # exponential of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.exp())
        Values:1.0, Derivatives:[1. 0.]

        """

        # compute the value and derivative of the exponential function for any input
        f = np.exp(self.val)
        f_prime = np.exp(self.val)
        return val_derv(f, self.derv * f_prime)

    def sin(self):
        """
        method to compute the value and derivative of the sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the sine function

        Examples
        --------
        # sine of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.sin())
        Values:0.0, Derivatives:1.0

        # sine of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.sin())
        Values:0.0, Derivatives:[1. 0.]

        """

        # compute the value and derivative of the sine function for any input
        f = np.sin(self.val)
        f_prime = np.cos(self.val)
        return val_derv(f, self.derv * f_prime)

    def cos(self):
        """
        method to compute the value and derivative of the cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the cosine function

        Examples
        --------
        # cosine of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.cos())
        Values:1.0, Derivatives:-0.0

        # cosine of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.cos())
        Values:1.0, Derivatives:[-0. -0.]

        """

        # compute the value and derivative of the cosine function for any input
        f = np.cos(self.val)
        f_prime = - np.sin(self.val)
        return val_derv(f, self.derv * f_prime)

    def tan(self):
        """
        method to compute the value and derivative of the tangent function

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
        # tangent of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.tan())
        Values:0.0, Derivatives:1.0

        # tangent of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.tan())
        Values:0.0, Derivatives:[1. 0.]

        # ValueError if input is an odd multiple of pi/2
        >>> x = val_derv(np.pi / 2, 1)
        >>> print(x.tan())
        ValueError: ERROR: Input to tan should not be an odd mutiple of pi/2

        """

        # ensure the user does not input an odd multiple of pi divided by 2
        if (self.val / (np.pi / 2)) % 2 == 1:
            raise ValueError("ERROR: Input to tan should not be an odd mutiple of pi/2")

        # compute the value and derivative of the tangent function for a valid input
        f = np.tan(self.val)
        f_prime = 1 / np.cos(self.val) ** 2
        return val_derv(f, self.derv * f_prime)

    def sinh(self):
        """
        method to compute the value and derivative of the hyperbolic sine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic sine function

        Examples
        --------
        # hyperbolic sine of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.sinh())
        Values:0.0, Derivatives:1.0

        # hyperbolic sine of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.sinh())
        Values:0.0, Derivatives:[1. 0.]

        """

        # compute the value and derivative of the hyperbolic sine for any input
        f = np.sinh(self.val)
        f_prime = np.cosh(self.val)
        return val_derv(f, self.derv * f_prime)

    def cosh(self):
        """
        method to compute the value and derivative of the hyperbolic cosine function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic cosine function

        Examples
        --------
        # hyperbolic cosine of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.cosh())
        Values:1.0, Derivatives:0.0

        # hyperbolic cosine of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.cosh())
        Values:1.0, Derivatives:[0. 0.]

        """

        # compute the value and derivative of the hyperbolic cosine function for any input
        f = np.cosh(self.val)
        f_prime = np.sinh(self.val)
        return val_derv(f, self.derv * f_prime)

    def tanh(self):
        """
        method to compute the value and derivative of the hyperbolic tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the hyperbolic tangent function

        Examples
        --------
        # hyperbolic tangent of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.tanh())
        Values:0.0, Derivatives:1.0

        # hyperbolic tangent of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.tanh())
        Values:0.0, Derivatives:[1. 0.]

        """

        # compute the value and derivative of the hyperbolic tangent function for any input
        f = np.tanh(self.val)
        f_prime = 1 / (np.cosh(self.val) ** 2)
        return val_derv(f, self.derv * f_prime)

    def arcsin(self):
        """
        method to compute the value and derivative of the inverse sine function

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
        # inverse sine of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.arcsin())
        Values:0.0, Derivatives:1.0

        # inverse sine of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.arcsin())
        Values:0.0, Derivatives:[1. 0.]

        # ValueError for input outside of the interval -1 to 1
        >>> x = val_derv(2, 1)
        >>> print(x.arcsin())
        ValueError: ERROR: Input to arcsin() should be between -1 and 1

        """

        # ensure the user passes in an input between -1 and 1
        if -1 >= self.val or self.val >= 1:
            raise ValueError("ERROR: Input to arcsin() should be between -1 and 1")
        # compute the value and derivative of the inverse sine function for a valid input
        f = np.arcsin(self.val)
        f_prime = 1 / (1 - self.val ** 2) ** 0.5
        return val_derv(f, self.derv * f_prime)

    def arccos(self):
        """
        method to compute the value and derivative of the inverse cosine function

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
        # inverse cosine of variable with scalar derivative
        >>> x = val_derv(0, 1)
        >>> print(x.arccos())
        Values:1.5707963267948966, Derivatives:-1.0

        # inverse cosine of variable with vector derivative
        >>> x = val_derv(0, np.array([1, 0]))
        >>> print(x.arccos())
        Values:1.5707963267948966, Derivatives:[-1. -0.]

        # ValueError for input outside of the interval -1 to 1
        >>> x = val_derv(2, 1)
        >>> print(x.arccos())
        ValueError: ERROR: Input to arccos() should be between -1 and 1

        """

        # ensure the user passes in an input between -1 and 1
        if -1 >= self.val or self.val >= 1:
            raise ValueError("ERROR: Input to arccos() should be between -1 and 1")
        # compute the value and derivative of the inverse cosine function for a valid input
        f = np.arccos(self.val)
        f_prime = - 1 / (1 - self.val ** 2) ** 0.5
        return val_derv(f, self.derv * f_prime)

    def arctan(self):
        """
        method to compute the value and derivative of the inverse tangent function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the inverse tangent function

        Examples
        --------
        # inverse tangent of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(x.arctan())
        Values:0.7853981633974483, Derivatives:0.5

        # inverse tangent of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(x.arctan())
        Values:0.7853981633974483, Derivatives:[0.5 0. ]

        """

        # compute the value and derivative of the inverse tangent function for a valid input
        f = np.arctan(self.val)
        f_prime = 1 / (1 + self.val ** 2)
        return val_derv(f, self.derv * f_prime)

    def logistic(self):
        """
        method to compute the value and derivative of the logistic function

        Parameters
        ----------
        None

        Returns
        -------
        A val_derv object that contains the value and derivative of the logistic function

        Examples
        --------
        # logistic of variable with scalar derivative
        >>> x = val_derv(1, 1)
        >>> print(x.logistic())
        Values:0.7310585786300049, Derivatives:0.19661193324148188

        # logistic of variable with vector derivative
        >>> x = val_derv(1, np.array([1, 0]))
        >>> print(x.logistic())
        Values:0.7310585786300049, Derivatives:[0.19661193 0. ]

        """

        return 1 / ((-self).exp() + 1)

