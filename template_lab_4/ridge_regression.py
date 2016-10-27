# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import compute_mse


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    k=len(y)
    l=tx.shape[1]
    
    #LA=lamb*np.eye(l,l)
    LA=lamb*2*k*np.eye(l,l)
    LA[0,0]=0

    gram=np.dot(np.transpose(tx),tx)+LA
    mat=np.dot(np.transpose(tx),y)
    weight=np.linalg.solve(gram,mat)
    mse= compute_mse(y,tx,weight)
    return weight, mse
    # ***************************************************
 
