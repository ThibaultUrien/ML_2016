# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    k=len(y)
    indices= np.random.permutation(np.arange(k))
    rd_x=x[indices]
    rd_y=y[indices]
   
    split=int(ratio*k)
    x1, x2 = rd_x[:split], rd_x[split:]
    y1, y2 = rd_y[:split], rd_y[split:]
    return x1, y1, x2, y2
    
