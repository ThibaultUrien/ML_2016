# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    gram=np.dot(np.transpose(tx),tx)
    im=np.dot(np.transpose(tx),y)
    opt_w=np.linalg.solve(gram,im)
    mse= compute_mse(y,tx,opt_w)
    return opt_w, mse
    
    # returns mse, and optimal weights
    # ***************************************************
   
