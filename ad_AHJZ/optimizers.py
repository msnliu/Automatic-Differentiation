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
    print('ADAM:',adam_optimizer(x, f_x, 1000)) 
    print()
    print('NADAM:',NADAM(x, f_x, 1000)) 
    print() 
    print('NAG:',NAG(x, f_x, 1000)) 
    print()
    print('RMSprop:', RMSprop(x, f_x, 1000)) 
    print()
    print('momentum:', momentum(x, f_x, 1000)) 
    print()
    print('BFGS:', BFGS(x, f_x, 1000)) 
    print()
    print('broyden:', broyden(x, f_x, 1000)) 
    print()

x = 3
f_x = lambda x: (x - 1)**2

def newtonScalar(x,f_x,num_iter, epsilon):
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
