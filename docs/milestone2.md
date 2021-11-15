*Describe the problem the software solves and why it’s important to
solve that problem.*

-   Our software package, *ad-AHJZ*, computes gradients by leveraging
    the technique of automatic differentiation. Before we can understand
    automatic differentiation, we must first describe and motivate the
    importance of differentiation itself. Derivatives are vital to
    quantifying the change that’s occurring over a relationship between
    multiple factors. Finding the derivative of a function measures the
    sensitivity to change of a function value with respect to a change
    in its input argument. Derivatives generalize across multiple
    scenarios and are well defined for both scalar inputs and outputs,
    as well as vector inputs and outputs. Derivatives are not only
    essential in calculus applications like numerically solving
    differential equations and optimizing and solving linear systems,
    but are useful in many real world, scientific settings. For example,
    in finance they help analyze the change regarding the profit/loss
    for a business or finding the minimum amount of material to
    construct a building. In physics, they help calculate the speed and
    distance of a moving object. Derivatives are crucial to
    understanding how such relationships move and change.

-   To perform differentiation, two different approaches are solving the
    task symbolically or numerically computing the derivatives. Symbolic
    differentation yields accurate answers, however depending on the
    complexity of the function, it could be expensive to evaluate and
    result in inefficient code. On the other hand, numerically computing
    derivatives is less expensive, however it suffers from potential
    issues with numerical stability and a loss of accuracy.

-   Our software package, *ad-AHJZ*, overcomes the shortcomings of both
    the symbolic and numerical approach. Our package uses automatic
    differentiation which is less costly than symbolic differentiation,
    but evaluates derivatives at machine precision. The technique
    leverages both forward mode and backward mode and evaluates each
    step with the results of previous computations or values. As a
    result of this, automatic differentiation avoids finding the entire
    analytical expresssion to compute the derivative and is hence
    iteratively evaluating a gradient based on input values. Thus, based
    on these key advantages, our library implements and performs forward
    mode automatic differentiation to efficiently and accurately compute
    derivatives.

*Describe (briefly) the mathematical background and concepts as you see
fit.*

The underlying motivation of automatic differentiation is the Chain Rule
that enables us to decompose a complex derivative into a set of
derivatives involving elementary functions of which we know explicit
forms.

We will first introduce the case of 1-D input and generalize it to
multidimensional inputs.

*One-dimensional (scalar) Input*: Suppose we have a function \$ f(y(t))
\$ and we want to compute the derivative of \$ f \$ with respect to \$ t
\$. This derivative is given by:

$$\begin{align}
\frac{\partial f}{\partial t} = \frac{\partial f}{\partial y} \frac{\partial y}{\partial t}\\
\end{align}$$

Before introducing vector inputs, let’s first take a look at the
gradient operator \$ \$

That is, for \$ y \^{n} \$, its gradient \$ y \^{n} \^{n}\$ is defined
at the point \$ x = (x\_1, …, x\_n) \$ in n-dimensional space as the
vector:

$$\begin{align}
\nabla y(x) =
\begin{bmatrix}
{\frac {\partial y}{\partial x_{1}}}(x)
\\
\vdots 
\\
{\frac {\partial y}{\partial x_{n}}}(x)
\end{bmatrix}
\end{align}$$

*Multi-dimensional (vector) Inputs*: Suppose we have a function \$
f(y\_1(x), …, y\_n(x)) \$ and we want to compute the derivative of \$ f
\$ with respect to \$ x \$. This derivative is given by:

$$\begin{align}
\nabla f_x = \sum_{i=1}^n \frac{\partial f}{\partial y_i} \nabla y_i(x)\\
\end{align}$$

We will introduce direction vector \$ p \$ later to retrieve the
derivative with respect to each \$ y\_i \$.

The Jacobian-vector product is equivalent to the tangent trace in
direction \$ p \$ if we input the same direction vector $p$:

\$ D\_p v \$ = \$ J p \$

Seed vectors provide an efficient way to retrieve every element in a
Jacobian matrix and also recover the full Jacobian in high dimensions.

*Scenario:* Seed vectors often come into play when we want to find \$
<span>x~j~</span> \$, which corresponds to the $i, j$ element of the
Jacobian matrix.

*Procedure:* In high dimension automatic differentiation, we will apply
seed vectors at the end of the evaluation trace where we have
recursively calculated the explicit forms of tangent trace of \$ f\_i
\$s and then multiply each of them by the indicator vector \$ p\_j \$
where the \$ j \$-th element of the \$ p \$ vector is 1.

*Definition*: Suppose x = \$ \\begin{bmatrix} {x\_1} \\ \\ {x\_m}
\\end{bmatrix} \$, we defined \$ v\_{k - m} = x\_k \$ for \$ k = 1, 2,
…, m \$ in the evaluation trace.

*Motivation*: The evaluation trace introduces intermediate results \$
v\_{k-m} \$ of elementary operations to track the differentiation.

Consider the function \$ f(x):\^2 \$:

\$ f(x) = log(x\_1) + sin(x\_1 + x\_2) \$

We want to evaluate the gradient \$ f \$ at the point \$ x =

7\
4

\$. Computing the gradient manually:

\$ f =

<span>x~1~</span>\
 <span>x~2~</span>

= \\begin{bmatrix} <span>x~1~</span> + (x\_1 + x\_2) \\ (x\_1 + x\_2)
\\end{bmatrix} =

<span>7</span> + (11)\
(11)

\$

[]<span>@llll@</span>

[b]<span>0.22</span>Forward primal trace

&

[b]<span>0.22</span>Forward tangent trace

&

[b]<span>0.22</span>Pass with p = $[0, 1]^T$

&

[b]<span>0.22</span>Pass with p = $[1, 0]^T$

[t]<span>0.22</span>$v_{-1} = x_1$

&

[t]<span>0.22</span>\$ p\_1 \$

&

[t]<span>0.22</span>1

&

[t]<span>0.22</span>0

[t]<span>0.22</span>$v_{0} = x_2$

&

[t]<span>0.22</span>\$ p\_2 \$

&

[t]<span>0.22</span>0

&

[t]<span>0.22</span>1

[t]<span>0.22</span>$v_{1} = v_{-1} + v_0$

&

[t]<span>0.22</span>\$ D\_p v\_{-1} + D\_p v\_0 \$

&

[t]<span>0.22</span>1

&

[t]<span>0.22</span>1

[t]<span>0.22</span>$v_{2} = sin(v_1)$

&

[t]<span>0.22</span>\$ (v\_1) D\_p v\_1 \$

&

[t]<span>0.22</span>\$ (11) \$

&

[t]<span>0.22</span>\$ (11) \$

[t]<span>0.22</span>$v_{3} = log(v_{-1})$

&

[t]<span>0.22</span>\$ <span>v~-1~</span> D\_p v\_{-1} \$

&

[t]<span>0.22</span>\$ <span>7</span> \$

&

[t]<span>0.22</span>0

[t]<span>0.22</span>\$v\_{4} = v\_3 + v\_2 \$

&

[t]<span>0.22</span>\$ D\_p v\_{3} + D\_p v\_2 \$

&

[t]<span>0.22</span>\$ <span>7</span> + (11) \$

&

[t]<span>0.22</span>\$ (11) \$

$D_p v_{-1} = \nabla v_{-1}^T p = (\frac {\partial v_{-1}} {\partial x_1} \nabla x_{1})^T p = (\nabla x_{1})^T p = p_1$

$D_p v_{0} = \nabla v_{0}^T p = (\frac {\partial v_{0}} {\partial x_2} \nabla x_{2})^T p = (\nabla x_{2})^T p = p_2$

$D_p v_{1} = \nabla v_{1}^T p = (\frac {\partial v_{1}} {\partial v_{-1}} \nabla v_{-1} + \frac {\partial v_{1}}{\partial v_{0}} \nabla v_{0})^T p = (\nabla v_{-1} + \nabla v_0)^T p = D_p v_{-1} + D_p v_0$

$D_p v_{2} = \nabla v_{2}^T p = (\frac {\partial v_{2}} {\partial v_{1}} \nabla v_1)^T p = \cos(v_1) (\nabla v_1)^T p = \cos(v_1) D_p v_1$

$D_p v_{3} = \nabla v_{3}^T p = (\frac {\partial v_{3}} {\partial v_{-1}} \nabla v_{-1})^T p = \frac {1} {v_{-1}} (\nabla v_{-1})^T p = \frac {1} {v_{-1}} D_p v_{-1}$

$D_p v_{4} = \nabla v_{4}^T p = (\frac {\partial v_{4}} {\partial v_3} \nabla v_{3} + \frac {\partial v_{4}}{\partial v_{2}} \nabla v_{2})^T p = (\nabla v_{3} + \nabla v_2)^T p = D_p v_{3} + D_p v_2$

We have connected each \$ v\_{k-m} \$ to a node in a graph for a
visualization of the ordering of operations.

From the above example, its computational graph is given by:

![computational\_graph.png](attachment:computational_graph.png)

Let’s generalize our findings:

From the table, we retrieved a pattern as below:

$$D_p v_j = (\nabla v_j)^T p = (\sum_{i < j} \frac{\partial{v_j}} {\partial{v_i}} \nabla v_i)^T p = \sum_{i < j} \frac{\partial{v_j}} {\partial{v_i}} (\nabla v_i)^T p = \sum_{i < j} \frac{\partial{v_j}} {\partial{v_i}} D_p v_i$$

*Higher dimension*: We recursively apply the same technique introduced
above to each entry of the vector valued function *f*.

Forward mode is efficient in the sense that it does not need to store
the parent node, which is different from reverse mode (see below) where
the whole computational graph must be stored.

The mechanism of reverse mode is defined as the following:

*Step 1:* Calculate \$ <span>v~j~</span> \$

*Step 2:* Calculate \$ <span>v~i~</span> \$ where \$ v\_i \$ is the
immediate predecessor of \$ v\_j \$

*Step 3:* Multiply the result obtained in step 1 and step 2, which
results in the following: \$ <span>v~j~</span> <span>v~i~</span> \$

*Naively*: We define a dual number \$ d\_i = v\_i + \_i \$ where \$ \_i
= D\_p v\_i \$ that satisfies \$ \^2 = 0 \$

A \$ k \$-th differentiable function \$ f \$ can be written as:

\$ f(d\_i) = f(v\_i + \_i) = f(v\_i) + f’(v\_i) \_i + <span>2!</span>
\_i\^2 + … + <span>k!</span> (- v\_i)\^k \$ for some \$ (v\_i, v\_i +
\_i) \$ by Taylor expansion.

Now we substitute the definition of \$ \_i \$ back into the above
expansion and use the fact that all higher terms go to 0 assuming \$ \^2
= 0 \$. We will have the following:

\$ f(d\_i) = f(v\_i) + f’(v\_i) D\_p v\_i \$

*Advantage*: Operations on Dual Number pertain to the form of Taylor
expansion, which makes the implementation easier to retrive the value
and derivative.

Consider the following example: $$\begin{align}
d_i &= v_i + D_p v_i \epsilon \\ 
f(d_i) &= d_i^2 = v_i^2 + 2 v_i D_p v_i \epsilon + D_p v_i^2 \epsilon^2 = v_i^2 + 2 v_i D_p v_i \epsilon \\
\end{align}$$

where \$ v\_i\^2 \$ refers to the value and \$ 2 v\_i D\_p v\_i \$
refers to the derivative.

More specifically, \$ v\_i\^2 \$ corresponds to \$ f(v\_i) \$, \$ 2 v\_i
\$ corresponds to \$ f’(v\_i) \$, and \$ D\_p v\_i \$ is just \$ D\_p
v\_i \$.

*How do you envision that a user will interact with your package? What
should they import? How can they instantiate AD objects?*

-   1a. User can install the package and its dependencies using the
    virtual environment venv:

[] <span><span>  /.virtualenvs</span></span>
<span><span>python3</span></span><span><span> -m venv
 /.virtualenvs/env~n~ame</span></span>
<span><span>source</span></span><span><span>
 /.virtualenvs/env~n~ame/bin/activate</span></span>
<span><span>python3</span></span><span><span> -m pip install
ad-AHJZ</span></span> <span><span>python3</span></span><span><span> -m
pip install -r requirements.txt</span></span>
<span><span>echo</span></span> <span><span>.py</span></span>

-   1b. User can install the package and its dependencies using the
    virtual environment conda:

[] <span><span>cd</span></span>
<span><span>conda</span></span><span><span> create -n
</span></span><span><span> python=3.7 anaconda</span></span>
<span><span>source</span></span><span><span> activate
env~n~ame</span></span> <span><span>python3</span></span><span><span> -m
pip install ad-AHJZ</span></span>
<span><span>python3</span></span><span><span> -m pip install -r
requirements.txt</span></span> <span><span>echo</span></span>
<span><span>.py</span></span>

-   2a. User imports package into the desired python file with the
    following line:

[] <span><span>from</span></span><span><span> ad~A~HJZ
</span></span><span><span>import</span></span><span><span> foward~m~ode,
</span></span>

-   2b. User imports numpy into the desired python file with the
    following line:

[] <span><span>import</span></span><span><span> numpy
</span></span><span><span>as</span></span><span><span> np</span></span>

-   3a. Using the class forward\_mode() create an automatic
    differentiation object that can use either a scalar or vector input
    to obtain both the function value and derivative. Below are examples
    using a scalar input and a vector input:

-   3b. Example of foward\_mode() using a scalar input:

[] <span><span>x </span></span> <span><span>f~x~ </span></span>
<span><span> x: np.sin(x) </span></span> <span><span> x</span></span>
<span><span>fm </span></span><span><span> forward~m~ode(x,
f~x~)</span></span> <span><span>x, x~d~er </span></span><span><span>
fm.get~f~unction~v~alue~a~nd~j~acobian()</span></span>
<span><span>print</span></span><span><span>(x, x~d~er)</span></span>
<span><span> [</span></span><span><span>]</span></span>
<span><span>x~v~alue </span></span><span><span>
fm.get~f~unction~v~alue()</span></span>
<span><span>print</span></span><span><span>(x~v~alue)</span></span>
<span><span>x~d~erivative </span></span><span><span>
fm.get~j~acobian()</span></span>
<span><span>print</span></span><span><span>(x~d~erivative)</span></span>
<span><span> [</span></span><span><span>]</span></span>

-   3c. Example of foward\_mode() using a vector input:

[] <span><span>multi~i~nput </span></span><span><span>
[</span></span><span><span>, </span></span><span><span>]</span></span>
<span><span>f~x~y </span></span> <span><span> x, y: np.sin(x)
</span></span> <span><span> y</span></span> <span><span>fm
</span></span><span><span> forward~m~ode(multi~i~nput,
f~x~y)</span></span> <span><span>multi~x~y, multi~x~y~d~er
</span></span><span><span>
fm.get~f~unction~v~alue~a~nd~j~acobian()</span></span>
<span><span>print</span></span><span><span>(multi~x~y,
multi~x~y~d~er)</span></span> <span><span> [</span></span> <span><span>
]</span></span> <span><span>multi~x~y~v~alue </span></span><span><span>
fm.get~f~unction~v~alue()</span></span>
<span><span>print</span></span><span><span>(multi~x~y~v~alue)</span></span>
<span><span>multi~x~y~d~erivative </span></span><span><span>
fm.get~j~acobian()</span></span>
<span><span>print</span></span><span><span>(multi~x~y~d~erivative)</span></span>
<span><span> [</span></span> <span><span> ]</span></span>

*Discuss how you plan on organizing your software package.*

-   1a. We include our project directory structure in the image below.
    Our package is called *ad-AHJZ*, where our code for automatic
    differentiation lies within “ad\_AHJZ”, our milestone documentation
    lies within “docs”, all unit testing files are located in “testing”,
    and the root of the directory holds our readme.md, license,
    .gitignore, .coveragerc, codecov.yml, setup.cfg, setup.py, and
    requirements.txt file.

-   1b. Directory structure layout:

-   2a. *val\_derv.py*: This file contains the class definition of a
    value/derivative object. It contains methods to initialize the
    object, set and get the function and derivative value of the object,
    and overload elementary operations. Specifically, we overload
    addition, multiplication, division, negation, power, reverse
    addition, reverse subtraction, reverse multiplication, and reverse
    division. Finally, we include elementary functions on these objects
    including ‘sqrt’, ‘log’, ‘exp’, ‘sin’, ‘cos’, ‘tan’, ‘sinh’, ‘cosh’,
    ‘tanh’, ‘arcsin’, ‘arccos’, and ‘arctan’. This is not a file which
    the user will interact with.

-   2b. *forward\_mode.py*: This file contains the class definition to
    perform forward mode automatic differentiation. This is the module
    which the user will interact with to compute function values and
    derivatives using forward mode. Specifically, the user will create
    forward mode objects using the function they are interested in
    computing the derivative of and the point or vector at which to
    evaluate the function at. Next, after initialization, they can make
    use of get\_function\_value() to retrieve function values,
    get\_jacobian to retrieve derivative values, and
    get\_function\_value\_jacobian() to retrieve both the function and
    derivative values.

-   2c. \_\_*init.py\_\_*: This file contains information relevant to
    how each of the modules associated with our package ad-AHJZ interact
    with one another.

-   2d . *reverse\_mode.py*: This file (once implemented) will contain
    the class definition to perform reverse mode automatic
    differentiation. This is the module which the user will interact
    with to compute function values and derivatives using reverse mode.
    Specifically, the user will create forward mode objects using the
    function they are interested in computing the derivative of and the
    point or vector at which to evaluate the function at. Next, after
    initialization, the user will call the methods on this objects to
    retrieve function and derivative values.

-   3a. The test suite live in the “testing” directory which is a
    subdirectory found off the root directory (see 1. Directory
    Structure). The “testing” directory contains all unit tests and
    integration tests.

-   3b. Our testing suite is built using Python’s unittest framework. We
    have two files for testing, which are test\_val\_derv.py and
    test\_forward\_mode.py. The first file tests scalar inputs for
    val\_derv.py to ensure all overloaded operations and elementary
    functions are implemented correctly and the second file tests
    forward\_mode.py to ensure the automatic differentiation is
    performed correctly in terms of computing function values and
    derivatives. We run our tests by running “coverage run  -m unittest
    discover -s tests/” in the root directory.

-   3c. To ensure our testing procedure has complete code coverage, we
    leverage CodeCov. CodeCov enables us to quickly understand which
    lines are being executed in our test cases. We directly upload our
    coverage reports to CodeCov through the use of a bash script and the
    .coveragerc, coverage, and codecov.yml files.

-   4a. Our package is distributed via PyPI. We have uploaded the
    package to PyPI using the setup.py and setup.cfg files which contain
    relevant information about our package as well as the version
    number, associated dependencices, and the license.

-   4b. A user can install our package by creating a virtual environment
    as shown in *Installing the package* under the “ How to Use ad-AHJZ”
    heading earlier. Once a virtual environment has been created, the
    user can install our package by running the following lines:

[] <span><span>python3</span></span><span><span> -m pip install
ad-AHJZ</span></span> <span><span>python3</span></span><span><span> -m
pip install -r requirements.txt</span></span>

-   4c: After installing our package, a user can import it into their
    desired python file and use it by including the following two lines
    at the top of their file:

[] <span><span>from</span></span><span><span> ad~A~HJZ
</span></span><span><span>import</span></span><span><span> foward~m~ode,
</span></span> <span><span>import</span></span><span><span> numpy
</span></span><span><span>as</span></span><span><span> np</span></span>

-   5a. The only library dependency our package relies on is numpy. We
    designed our software in this manner to ensure that we are not
    creating multiple external dependencies and thereby increase our
    software’s reliability.

*Discuss how you plan on implementing the forward mode of automatic
differentiation.*

-   1a. Our primary core data structure is the numpy array, which we use
    to store both the variable list and the function list. Then using
    the methods within the forward\_method class we compute the jacobian
    and function value storing those values or arrays, depending on the
    input, in a tuple.

-   2a. *Val Derv:* The class that creates our val\_derv object. This
    object has two attributes: the value and the derivative seed, which
    can be defined at instantiation. This object will be used with the
    elementary function methods to calculate the value, and the dual
    number at a particular state of the primal or tangent trace.

-   2b. *Forward Mode:* The class that creates a forward\_mode object.
    This object has two attributes: the variable list and the function
    list, which can be defined at instantiation. Both attributes can be
    in either the scalar or vector form, and will be used to find either
    the function value, the jacobian, or both.

-   2c. *Reverse Mode* (extension module): The class that creates a
    reverse\_mode object. This object has two attributes: the variable
    list and the function list, which can be defined at instantiation.
    Both attributes can be in either the scalar or vector form, and will
    be used to find either the function value, the jacobian, or both.

-   3a. *Val Derv*

    -   **init**: Constructor for the val\_derv class.

    -   **repr**: Operator overloading for val\_derv object string
        representations

    -   @property val : Gets the val attribute of val\_derv object

    -   @property derv : Gets the derv attribute of val\_derv object

    -   @val.setter val : Sets the val attribute of val\_derv object

    -   @derv.setter derv : Sets the derv attribute of val\_derv object

    -   **add**: Compute the value and derivative of the addition
        operation

    -   **mul**: Compute the value and derivative of the multiplication
        operation

    -   **truediv**: Compute the value and derivative of the division
        operation

    -   **neg**: Compute the value and derivative of the negation
        operation

    -   **pow**: Compute the value and derivative of the power operation

    -   **radd**: Compute the value and derivative of the addition
        operation

    -   **rsub**: Compute the value and derivative of the subtraction
        operation

    -   **rmul**: Compute the value and derivative of the multiplication
        operation

    -   **rtruediv**: Compute the value and derivative of the division
        operation

    -   **rpow**: Compute the value and derivative of the power
        operation

    -   sqrt: Compute the value and derivative of the square root
        function

    -   log: Compute the value and derivative of logarithmic function
        (Default logarithmic base is None)

    -   exp: Compute the value and derivative of exponential function

    -   sin: Compute the value and derivative of the sine function

    -   cos: Compute the value and derivative of the cosine function

    -   tan: Compute the value and derivative of the tangent function

    -   sinh: Compute the value and derivative of the hyperbolic sine
        function

    -   cosh: Compute the value and derivative of the hyperbolic cosine
        function

    -   tanh: Compute the value and derivative of the hyperbolic tangent
        function

    -   arcsin: Compute the value and derivative of the inverse sine
        function

    -   arccos: Compute the value and derivative of the inverse cosine
        function

    -   arctan: Compute the value and derivative of the inverse tangent
        function

-   3b. *Forward Mode*

    -   **init**: Constructor for the forward\_mode class.

    -   get\_function\_value: Extracts the function value from the
        function ‘get\_function\_value\_and\_jacobian’

    -   get\_jacobian: Extracts the jacobian matrix from the function
        ‘get\_function\_value\_and\_jacobian’

    -   get\_function\_value\_and\_jacobian: Calculates the function
        value and jacobian of a user input function

-   3c. *Reverse Mode (Potential Methods)*

    -   **init**: Constructor for the forward\_mode class.

    -   get\_function\_value: Extracts the function value from the
        function ‘get\_function\_value\_and\_jacobian’

    -   get\_jacobian: Extracts the jacobian matrix from the function
        ‘get\_function\_value\_and\_jacobian’

    -   get\_function\_value\_and\_jacobian: Calculates the function
        value and jacobian of a user input function

-   4a. To be viewed as a near stand alone software package, to improve
    adoption, and increase efficiency, we chose to only employ a single
    external library, numpy. We’ve used the numpy library to create our
    data structure for the computational graph and perform computations
    outside of those we created in our val\_derv class.

-   5a. As listed above, within the val\_derv class we’ve overloaded the
    simple arithmetic functions (addition, subtraction, multiplication,
    and power) to calculate both the value and the dual number. We’ve
    also defined our own elementary functions, such as sin(x) and
    sqrt(x) to also compute the value and the derivative. This module
    will generalize each of the functions in order to handle both scalar
    and vector inputs. Each method will also indicate errors specific to
    the types of possible invalid inputs. The output will be a tuple of
    both the function value and the derivative, which will be used in
    both the foward\_mode and the reverse\_mode.\

-   5b. Below are examples of how we would implement *sin* and *sqrt*,
    both of which work with scalar or vector input *x* values, within
    the val\_derv class:

[] <span><span>x </span></span><span><span>
val~d~erv(</span></span><span><span>,
</span></span><span><span>)</span></span>
<span><span>print</span></span><span><span>(x.sqrt())</span></span>
<span><span>Values:</span></span><span><span>,
Derivatives:</span></span>

[] <span><span>x </span></span><span><span>
val~d~erv(</span></span><span><span>,
np.array([</span></span><span><span>,
</span></span><span><span>]))</span></span>
<span><span>print</span></span><span><span>(x.sqrt())</span></span>
<span><span>Values:</span></span><span><span>,
Derivatives:[</span></span> <span><span> ]</span></span>

[] <span><span>x </span></span><span><span>
val~d~erv(</span></span><span><span>,
</span></span><span><span>)</span></span>
<span><span>print</span></span><span><span>(x.sin())</span></span>
<span><span>Values:</span></span><span><span>,
Derivatives:</span></span>

[] <span><span>x </span></span><span><span>
val~d~erv(</span></span><span><span>,
np.array([</span></span><span><span>,
</span></span><span><span>]))</span></span>
<span><span>print</span></span><span><span>(x.sin())</span></span>
<span><span> Values:</span></span><span><span>,
Derivatives:[</span></span> <span><span>]</span></span>

-    

    each method not only calculates the value of the elmentary function
    using an input(either scalar or vector), but also claculates the
    derivative using dual numbers - add steps of the different
    attributes in each method for the val\_derv class overall

    -   forward\_mode() – add steps of different attributes in each
        method for overall class

-4) External dependencies - numpy - Elementary functions - list of
functions and their overall desciptions -5) This is similar to what you
did for Milestone 1, but now you’ve actually implemented it. - semi
progress reports of updates from ilestone 1 - and what to do next

-6) What aspects have you not implemented yet? What else do you plan on
implementing? - we have not implemented our addition features such as
reverse\_mode() and other elementary functions that a user may want to
use. We are also hoping to implement an optimized version of our
forward\_mode().

–\>

*Briefly motivate your license choice*

Our *ad-AHJZ* package is licensed under the GNU General Public License
v3.0. This free software license allows users to do just about anything
they want with our project, except distribute closed source versions.
This means that any improved versions of our package that individuals
seek to release must also be free software. We find it essential to
allow users to help each other share their bug fixes and improvements
with other users. Our hope is that users of this package continually
find ways to improve it and share these improvements within the broader
scientific community that uses automatic differentation.

*Discuss how you plan on expanding the automatic differentiation package
to include additional features.*

-   One key area which we would like to expand or implement next in our
    package is the support for more elementary functions and operations
    in val\_derv.py. Currently, we provide users with twenty-two options
    which overload basic arithmetic operations along with the
    trignometric functions. However, for advanced users that could use
    our package to solve complex computational problems, we believe that
    providing support for even more complex functions and operations
    could prove useful in providing generalizability.

-   Another key capability we would like to implement is providing
    reverse mode automatic differentiation. Providing a class analogous
    to forward mode, but which instead simulates reverse mode is
    important due to the computational efficiency we can provide our
    users with. Compared to forward mode, reverse mode has a
    significantly smaller arithmetic operation count for mappings of the
    form $f(x): R^{n} \rightarrow R^{m}$ when $n >> m$. Users that
    choose to use our library to tackle large-scale machine learning
    tasks would require efficient and reliable differentiation and so it
    is critical that our package provide users with this opton to carry
    out automatic differentiation using either method. We plan on
    creating a reverse mode class which will serve as our primary
    extension module for this project.

-   We would also like to implement an enhanced version of forward mode,
    time permitting. We believe that the reverse mode implementation
    mentioned above will certainly provide computational efficiency for
    users that want access to fast differentiation for mappings of the
    form $f(x): R^{n} \rightarrow R^{m}$ when $n >> m$, however
    optimizing the forward mode implementation could enable users to
    access fast differentiation for mappings which are of the form
    $f(x): R^{n} \rightarrow R^{m}$ when $m >> n$.

-   Finally, we would like to increase our testing suite. Currently, our
    testing suite only tests scalar inputs. However, we have already
    included functionality in our library to allow for both vector input
    and vector output. Hence, we would like to expand our testing suite
    to test these cases thoroughly.

-   After adding support for additional elementary functions and
    operations, our software wil change in two key ways. First off, the
    val\_derv.py file will contain additional instance methods
    pertaining to the new functions and operations we implement.
    Specifically, we will have as many new instance methods as
    additional elementary functions/operations we provide. Additionally,
    the testing suite file test\_val\_derv.py will contain additional
    test cases that perform unit testing on these new elementary
    functions and operations.

-   To implement the reverse mode capability, we will need to create a
    new module, called reverse\_mode.py. This module will contain the
    class definition for the reverse mode automatic differentiation and
    will be set-up similarly to the forward\_mode.py module. We note
    that a key implementation detail in the reverse mode will be the
    underlying data structure we use to represent the computational
    graph. After a lot of research, we realize that we need a data
    structure such as a dictionary. This will enable us to save time in
    retrieving stored operations as the complexity of our outputs
    increases. In addition to creating the new reverse mode module, we
    will need to add a new testing file which includes unit and
    integration tests for reverse mode.

-   If we were to optimize our forward mode implementation, we would
    have to change the underlying data structure which represents the
    computational graph. A Chain Map data structure would be used
    instead of our current implementation in order to decrease the
    number of repetitive calculations and increase efficiency as the
    input complexity increases.

-   The primary challenge we will face with implementing additional
    elementary functions and operations is dealing with the dual number
    cases for these additions. We have found that many of the elementary
    functions and operations are simple to implement in the real number
    case, however when dealing with dual numbers they can become quite
    complex and so special care must be taken in handling these cases.

-   The primary challenge in implementing reverse mode will be
    understanding the structural and methodological differences between
    forward and reverse mode. For example, before we can implement the
    reverse mode module, we need to understand if we can even make use
    of our code from forward mode as a starting point or if we will have
    to completely reimplement the reverse mode module from scratch.

-   The primary challenge we expect to fact with optimizing our forward
    mode implementation is in the use of a Chain Map data structure. We
    will have to understand, in detail, how this complex data structure
    works and look into how our current code implementation would work
    with such a data structure.

**2/2 Introduction:** Would have been nice to see more about why do we
care about derivatives anyways and why is undefined a solution compared
to other approaches?

*Response:* In the Introduction, we addressed these comments by
explaining the purpose of derivatives, expanding upon the real-world
applications of derivatives, and their generalizability across multiple
dimensions. We addressed this comment in the updated first bullet point
of the introduction section.

**2/2 Background:** Good start to the background. Going forward, I would
like to see more discussion on automatic differentiation. How do forward
mode and reverse mode work? I would also like to see more discussion on
what forward mode actually computes (Jacobian-vector product), the
“seed” vector, and the efficiency of forward mode.

*Response:* In the Background, we addressed these comments by adding
four new subsections which discuss the topics of reverse mode,
jacobian-vector product, seed vectors, and the efficiency of forward
mode. This new information is contained in Part 2, Part 3, Part 7, and
Part 8, in the background section.

**3/3 How to use:** Good Job!

Response: N/A

**2/2 Software Organization:** Nicely Done!

*Response: N/A*

**4/4 Implementation:** Classes and methods are very well
thought-through. It would be great if you could list all the elementary
operations that you will overload.

*Response:* In the Implementation, we addressed this comment by listing
all fourteen elementary operations which we plan on overloading. We
addressed this comment in the subsection “3a. Dual Numbers” under the
second bullet point.

**2/2 License:** Good Job!

*Response:* N/A
