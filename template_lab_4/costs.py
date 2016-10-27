# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

def compute_mse(y, tx, beta):
    """compute the loss by mse."""
    e = y - np.dot(tx,beta)
    mse = np.linalg.norm(e)**2 / (2 * len(e))
    return mse
