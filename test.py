from forward_mode import forward_mode
from val_derv import val_derv
import numpy as np
from optimizers import optimizer
x = 0.5
#x = val_derv(1,1)
f_x = lambda x: x.sin().exp() - (x**0.5).cos()* ((x.cos()**2 + x**2)**0.5).sin()
fm = forward_mode(x, f_x)
x, x_der = fm.get_function_value_and_jacobian()
print(x, x_der)
#print(exp(x))


#print(x.sin())
