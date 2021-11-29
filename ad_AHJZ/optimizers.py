#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 21:28:56 2021

@author: aditimemani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 16:17:01 2021

@author: aditimemani
"""


import numpy as np
from forward_mode import forward_mode 
import time 

x1 = 1
f_x1 = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
x2 = np.array([1,-1])
f_x2 = lambda x,y:x**3 +y**2
x3 = 2 
f_x3 = lambda x: (x-1)**2 +5 
x_list = [x1, x2, x3]
fx_list = [f_x1, f_x2, f_x3]



def adam_optimizer(x,f_x, num_iter, alpha = 0.01, beta1=.9, beta2=.999, epsilon = 1e-8 ):
    """
    Parameters
    ----------
    x: the variable input (can be in either scalar or vector form)
    f_x: the function you would like to obtain the minimum for
    num_iter: the numnber of interations
    alpha = 0.01: default learning rate or the gradiant descent
    beta1=.9: default value for the exponential decay of the first moment
    beta2=.999: default value for the exponential decay of the second moment
    epsilon = 1e-8: default value to assure that ZeroDivisionError is not raised

    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 1
    >>> f_x = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
    >>> adam_optimizer(x, f_x, 1000)
    (0.19765210151672363, 0.26172998379097046, array([0.94233316]))
    
    >>> x = np.array([1,-1])
    >>> f_x = lambda x,y:x**3 +y**2
    >>> adam_optimizer(x, f_x, 1000)
    (0.09849190711975098, 6.03886825409073e-06, array([1.82103595e-02, 1.81385270e-21]))
    
    >>> x = 2
    >>> f_x = lambda x: (x-1)**2 +5
    >>> adam_optimizer(x, f_x, 1000)
    (0.08600378036499023, 5.0, array([1.]))
    """
    start = time.time()
    if beta1 >=0 and beta1<1:
        if beta2 >=0 and beta2<1:
            mt = 0 
            vt = 0 
            t=0
            curr_val = x
            fm = forward_mode(x,f_x)
            val, x_der = fm.get_function_value(), fm.get_jacobian()
        
            for t in range(1,num_iter+1):
#                gt = fm.get_jacobian()
                mt = beta1*mt + (1-beta1)*x_der
                vt = beta2*vt +(1-beta2)*x_der**2
                mhat = mt/(1-beta1**t)
                vhat = vt/(1-beta2**t)
                variation =  alpha*mhat/(np.sqrt(vhat)+epsilon)
                curr_val = curr_val-variation
                fm = forward_mode(curr_val, f_x)
                val, x_der = fm.get_function_value(), fm.get_jacobian()
                
        else:
            raise ValueError("Beta Values must be within the range of [0,1)")
    else:
        raise ValueError("Beta Values must be within the range of [0,1)")
    end = time.time()
    opt_time = end - start
    return opt_time, val, curr_val
#print('ADAM:',adam_optimizer(x, f_x, 10000)) 
#print()   
    
    

def NADAM (x,f_x, num_iter, alpha = 0.01, beta1=.9, beta2=.999, epsilon = 1e-8 ):
    """
    Parameters
    ----------
    x: the variable input (can be in either scalar or vector form)
    f_x: the function you would like to obtain the minimum for
    num_iter: the numnber of interations
    alpha = 0.01: default learning rate or the gradiant descent
    beta1=.9: default value for the exponential decay of the first moment
    beta2=.999: default value for the exponential decay of the second moment
    epsilon = 1e-8: default value to assure that ZeroDivisionError is not raised

    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 1
    >>> f_x = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
    >>> NADAM(x, f_x, 1000)
    (0.17513608932495117, 0.26172998379097046, array([0.94233316]))
    
    >>> x = np.array([1,-1])
    >>> f_x = lambda x,y:x**3 +y**2
    >>> NADAM(x, f_x, 1000)
    (0.09782099723815918, 6.246798742956698e-06, array([ 1.8417012e-02, -2.7649170e-21]))
    
    >>> x = 2
    >>> f_x = lambda x: (x-1)**2 +5
    >>> NADAM(x, f_x, 1000)
    (0.07419586181640625, 5.0, array([1.]))
    """
    start = time.time()
    if beta1 >=0 and beta1<1:
        if beta2 >=0 and beta2<1:
            mt = 0 
            vt = 0 
            t=0
            curr_val = x
            fm = forward_mode(x,f_x)
            val, x_der = fm.get_function_value(), fm.get_jacobian()
            #mt is vt in NAG but changed to incorporate first and second past gradient moments
            for t in range(1,num_iter+1):
                mt = beta1*mt + (1-beta1)*x_der
                vt = beta2*vt +(1-beta2)*x_der**2
                mhat = mt/(1-beta1**t)
                vhat = vt/(1-beta2**t)
                mvar = beta1*mhat +(1-beta1)*x_der #new variable added vs ADAM
                variation =  alpha*mvar/(np.sqrt(vhat+epsilon)) #changed from ADAM
                curr_val = curr_val-variation
                fm = forward_mode(curr_val, f_x)
                val, x_der = fm.get_function_value(), fm.get_jacobian()
                
        else:
            raise ValueError("Beta Values must be within the range of [0,1)")
    else:
        raise ValueError("Beta Values must be within the range of [0,1)")
    end = time.time()
    opt_time = end - start
    return opt_time, val, curr_val
#print('NADAM:',NADAM(x, f_x, 10000)) 
#print()
#nag uses only 1 past gradient moment 
def NAG(x,f_x, num_iter, alpha=.1, gamma=0.9, epsilon = 1e-8 ):
    """
    Parameters
    ----------
    x: the variable input (can be in either scalar or vector form)
    f_x: the function you would like to obtain the minimum for
    num_iter: the numnber of interations
    alpha = 0.1: default learning rate or the gradiant descent
    gamma=.9: default value for the exponential decay
    epsilon = 1e-8: default value to assure that ZeroDivisionError is not raised

    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 1
    >>> f_x = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
    >>> NAG(x, f_x, 1000)
    (0.206679105758667, 0.26177237171374856, array([0.94615669]))
    
    >>> x = np.array([1,-1])
    >>> f_x = lambda x,y:x**3 +y**2
    >>> NAG(x, f_x, 1000)
    (0.09226107597351074, 0.14092469295850776, array([ 0.38994091, -0.28571429]))
    
    >>> x = 2
    >>> f_x = lambda x: (x-1)**2 +5
    >>> NAG(x, f_x, 1000)
    (0.07019376754760742, 5.081632653061225, array([1.28571429]))
    """
    start = time.time()
    if gamma >=0 and gamma<1:
        vt = 0
        t = 0
        curr_val = x
        for t in range(1,num_iter+1):
            var = gamma*vt
            curr_val = curr_val
#            print('curr_val:',curr_val)
            J_input = curr_val-var
            fm = forward_mode(J_input, f_x)
            x_der = fm.get_jacobian()
            vt = var + alpha*x_der
            curr_val1 = curr_val-vt
#            print('curr_val1:',curr_val1)
            fm = forward_mode(curr_val1, f_x)
            val = fm.get_function_value()

    else:
        raise ValueError("Gamma Value must be within the range of [0,1)")
    end = time.time()
    opt_time = end - start
    return opt_time, val, curr_val1 
#print('NAG:',NAG(x, f_x, 1000)) 
#print()
#ii = np.arange(0.01, 0.11, 0.01)
#print(ii)
#jj = np.arange(0.6, 1, 0.1)
#print(jj)
#for i in ii: 
#    for j in jj:
#        print(NAG(x, f_x, 10000, i, j))  
#        print()

def RMSprop(x,f_x, num_iter, alpha = 0.01, beta=.9, epsilon = 1e-8 ):
    """
    Parameters
    ----------
    x: the variable input (can be in either scalar or vector form)
    f_x: the function you would like to obtain the minimum for
    num_iter: the numnber of interations
    alpha = 0.01: default learning rate or the gradiant descent
    beta =.9: default value for the exponential decay
    epsilon = 1e-8: default value to assure that ZeroDivisionError is not raised

    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 1
    >>> f_x = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
    >>> RMSprop(x, f_x, 1000)
    (0.3034090995788574, 0.2618028370373199, array([0.93730206]))
    
    >>> x = np.array([1,-1])
    >>> f_x = lambda x,y:x**3 +y**2
    >>> RMSprop(x, f_x, 1000)
    (0.08888602256774902, 2.4997500000081948e-05, array([ 4.34424616e-06, -4.99974999e-03]))
    
    >>> x = 2
    >>> f_x = lambda x: (x-1)**2 +5
    >>> RMSprop(x, f_x, 1000)
    (0.07326507568359375, 5.0, array([1.]))
    """
    start = time.time()
    if beta >=0 and beta<1:
        vt = 0 
        t=0
        curr_val = x
        fm = forward_mode(x,f_x)
        val, x_der = fm.get_function_value(), fm.get_jacobian()
    
        for t in range(1,num_iter+1):
            vt = beta*vt +(1-beta)*x_der**2
            variation =  alpha*x_der/(np.sqrt(vt+epsilon))
            curr_val = curr_val-variation
            fm = forward_mode(curr_val, f_x)
            val, x_der = fm.get_function_value(), fm.get_jacobian()
    else:
        raise ValueError("Beta Values must be within the range of [0,1)")
    end = time.time()
    opt_time = end - start
    return opt_time, val, curr_val

def momentum(x,f_x, num_iter, alpha = 0.01, beta=.9, epsilon = 1e-8):
    """
    Parameters
    ----------
    x: the variable input (can be in either scalar or vector form)
    f_x: the function you would like to obtain the minimum for
    num_iter: the numnber of interations
    alpha = 0.01: default learning rate or the gradiant descent
    beta=.9: default value for the exponential decay
    epsilon = 1e-8: default value to assure that ZeroDivisionError is not raised

    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 1
    >>> f_x = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
    >>> momentum(x, f_x, 1000)
    (0.24898195266723633, 0.26172998379097046, array([0.94233316]))
    
    >>> x = np.array([1,-1])
    >>> f_x = lambda x,y:x**3 +y**2
    >>> momentum(x, f_x, 1000)
    (0.08585095405578613, 2.7605629377339922e-05, array([ 3.02226506e-02, -3.30135704e-12]))
    
    >>> x = 2
    >>> f_x = lambda x: (x-1)**2 +5
    >>> momentum(x, f_x, 1000)
    (0.06420779228210449, 5.0, array([1.]))
    """
    start = time.time()
    if beta >=0 and beta<1:
            mt = 0 
            t=0
            curr_val = x
            fm = forward_mode(x,f_x)
            val, x_der = fm.get_function_value(), fm.get_jacobian()
            for t in range(1,num_iter+1):
#                gt = fm.get_jacobian()
                mt = beta*mt + (1-beta)*x_der
                variation =  alpha*mt
                curr_val = curr_val-variation
                fm = forward_mode(curr_val, f_x)
                val, x_der = fm.get_function_value(), fm.get_jacobian()
    else:
        raise ValueError("Beta Values must be within the range of [0,1)")
    end = time.time()
    opt_time = end - start
    return opt_time, val, curr_val 
def BFGS(x,f_x, num_iter, alpha = 0.01):
    """
    Parameters
    ----------
    x: the variable input (can be in either scalar or vector form)
    f_x: the function you would like to obtain the minimum for
    num_iter: the numnber of interations
    alpha = 0.01: default learning rate or the gradiant descent
    
    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 1
    >>> f_x = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
    >>> BFGS(x, f_x, 1000)
    (0.2875189781188965, 0.2617299838095016, array([0.94233569]))
    
    >>> x = np.array([1,-1])
    >>> f_x = lambda x,y:x**3 +y**2
    >>> BFGS(x, f_x, 1000)
    (0.12035322189331055, 2.864368146437936e-07, array([ 6.57872856e-03, -4.13716985e-05]))
    
    >>> x = 2
    >>> f_x = lambda x: (x-1)**2 +5
    >>> BFGS(x, f_x, 1000)
    ((0.08925986289978027, 5.000000001826295, array([1.00004274]))
    """
    start = time.time()
    if np.isscalar(x):
        n= 1
    else:
        n = len(x)
    curr_val = x
    curr_hessian = np.eye(n)
    fm = forward_mode(x,f_x)
    val, x_der = fm.get_function_value(), fm.get_jacobian()
    for t in range(1,num_iter+1):
        variation = -alpha*curr_hessian@x_der
        curr_val = curr_val + variation
        fm = forward_mode(curr_val, f_x)
        val, x_der1 = fm.get_function_value(), fm.get_jacobian()
        id1 = np.eye(n)
        yk = (x_der1-x_der).reshape(-1,1)
        xk = variation.reshape(-1,1)
        denom = yk.T@xk
        t1 = (id1-xk@yk.T/denom)
        t2 = (id1-yk@xk.T/denom)
        t3 = xk@xk.T/denom
        curr_hessian = t1@ curr_hessian@t2 +t3
        x_der =x_der1
    end = time.time()
    opt_time = end - start
    return opt_time, val, curr_val  

def broyden(x,f_x, num_iter, alpha = 0.01):
    """
    Parameters
    ----------
    x: the variable input (can be in either scalar or vector form)
    f_x: the function you would like to obtain the minimum for
    num_iter: the numnber of interations
    alpha = 0.01: default learning rate or the gradiant descent
    
    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 1
    >>> f_x = lambda x: (-1*np.log(x))+(np.exp(x)*x**4)/10
    >>> broyden(x, f_x, 1000)
    (0.2750861644744873, 0.2617299838095016, array([0.94233569]))
    
    >>> x = np.array([1,-1])
    >>> f_x = lambda x,y:x**3 +y**2
    >>> broyden(x, f_x, 1000)
    (0.1135101318359375, 3.060640040173604e-07, array([ 6.72566138e-03, -4.28010827e-05]))
    
    >>> x = 2
    >>> f_x = lambda x: (x-1)**2 +5
    >>> broyden(x, f_x, 1000)
    broyden: (0.0851430892944336, 5.000000001826295, array([1.00004274]))
    """
    start = time.time()
    if np.isscalar(x):
        n= 1
    else:
        n = len(x)
    curr_val = x
    curr_hessian = np.eye(n)
    fm = forward_mode(x,f_x)
    val, x_der = fm.get_function_value(), fm.get_jacobian()
    for t in range(1,num_iter+1):
        variation = -alpha*curr_hessian@x_der
        curr_val = curr_val + variation
        fm = forward_mode(curr_val, f_x)
        val, x_der1 = fm.get_function_value(), fm.get_jacobian()
        yk = (x_der1-x_der).reshape(-1,1)
        xk = variation.reshape(-1,1)
        broyden_numerator = (xk-curr_hessian@yk)@xk.T@curr_hessian
        broyden_denom = xk.T@curr_hessian@yk
        curr_hessian = curr_hessian + broyden_numerator/broyden_denom
        x_der =x_der1
    end = time.time()
    opt_time = end - start
    return opt_time, val, curr_val     
        

for x, f_x in zip (x_list, fx_list):
    print('x=',x)
    print('f_x=',f_x)
    print('ADAM:',adam_optimizer(x, f_x, 1000)) 
    print()
    print('x=',x)
    print('f_x=',f_x)
    print('NADAM:',NADAM(x, f_x, 1000)) 
    print() 
    print('x=',x)
    print('f_x=',f_x)
    print('NAG:',NAG(x, f_x, 1000)) 
    print()
    print('x=',x)
    print('f_x=',f_x)
    print('RMSprop:', RMSprop(x, f_x, 1000)) 
    print()
    print('x=',x)
    print('f_x=',f_x)
    print('momentum:', momentum(x, f_x, 1000)) 
    print()
    print('x=',x)
    print('f_x=',f_x)
    print('BFGS:', BFGS(x, f_x, 1000)) 
    print()
    print('x=',x)
    print('f_x=',f_x)
    print('broyden:', broyden(x, f_x, 1000)) 
    print()

x = 3
f_x = lambda x: (x - 1)**2

def newtonScalar(x,f_x,num_iter, epsilon=1e-6):
    """
    Parameters
    ----------
    x: the variable input (can ONLY be scalar)
    f_x: the function you would like to use
    num_iter: the numnber of interations
    epsilon = 1e-6: default value to assure that ZeroDivisionError is not raised

    Returns
    -------
    opt_time: The time it takes to run the optimizer in seconds
    val: the position of the minimum value
    curr_val: the minimum value (can be in either scalar or vector form)
    
    Examples
    --------
    >>> x = 3
    >>> f_x = lambda x: (x - 1)**2
    >>> newtonScalar(x, f_x, 100, 1e-6)
    (0.0013539791107177734, array([1.00000191]), 3.637978807091713e-12)
    """
    start = time.time()
    for n in range(1,num_iter+1):
        fm = forward_mode(x,f_x)
        val = fm.get_function_value()
        der = fm.get_jacobian()
        if der == 0:
            raise ValueError("Unable to find solution (zero derivative).")
        delta = val/der
        next_x = x - delta
        prev_x = x
        x = next_x
        if np.abs(delta) < epsilon:
            break

    root = prev_x
    end = time.time()
    opt_time = end - start
    return opt_time, root, val

print(newtonScalar(x, f_x, 100, 1e-6))
