# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
from costs import compute_loss_mse

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


            
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    e=y-np.dot(tx,w)
    grad=-np.dot(np.transpose(tx),e)/(len(y))
    return grad
 

def least_square_GD(y, tx, initial_w,gamma, max_iters): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        G=compute_gradient(y,tx,w)
        loss=compute_loss_mse(y,tx, w)
        # update w by gradient
        w=w-gamma*G
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
    print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws, losses[-1], ws[-1]

def least_square_SGD(y, tx, batch_size, initial_w, max_epochs, nouv_it, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    
    ws = [initial_w]
    losses = []
    w = initial_w;
    
    for n_iter in range(max_epochs):
        i=0
        print(i)
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # compute gradient and loss, store them
            if i==nouv_it:
                break
            i=i+1
            grad= compute_gradient(minibatch_y, minibatch_tx, w)
            w=w-gamma*grad
            ws.append(np.copy(w))
            loss=compute_loss_mse(y,tx,w)
            losses.append(loss)
           
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
                  bi=n_iter, ti=max_epochs - 1, l=loss))
      
    return losses, ws, losses[-1], ws[-1]

def least_squares(y, tx):
    """calculate the least squares solution."""
  
    # Computing optimal weights
    gram=np.dot(tx.T,tx)
    im=np.dot(tx.T,y)
    opt_w=np.linalg.solve(gram,im)
    
    # Computing mean square error 
    mse=compute_loss_mse(y,tx,opt_w)
    
    return opt_w, mse
    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    # Initializing useful parameters
    k=len(y)
    l=tx.shape[1]
    #LA=lambda_*np.eye(l,l)
    LA=lambda_*2*k*np.eye(l,l)
    LA[0,0]=0
    
    # Computing optimal weights
    gram=np.dot(tx.T,tx)+LA
    mat=np.dot(tx.T,y)
    weight=np.linalg.solve(gram,mat)
    
    # Computing mean square error
    mse= compute_loss_mse(y,tx,weight)
    
    return weight, mse

# Pour la logistique

def sigmoid(t):
    """apply sigmoid function on t."""

    return np.exp(t)/(1+np.exp(t))

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    
    #loss = y*np.log(sigmoid(np.dot(tx,w))) + (1-y)*np.log(1-sigmoid(np.dot(tx, w)))
    logistic_loss = np.log (1+np.exp(np.dot(tx,w)))-y*np.dot(tx,w)
    return np.sum(logistic_loss)

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""

    err= sigmoid(np.dot(tx,w))-y
    return np.dot(tx.T,err)

def calculate_logistic_hessian(y, tx, w):
    """return the hessian of the loss function."""

    sig=sigmoid(np.dot(tx,w))*(1-sigmoid(np.dot(tx,w)))

    S=np.diag(sig[:,0])

    return np.dot(np.dot(tx.T,S),tx)

# LOGISTIC REG AVEC UN GRADIENT
# UNE ITERATION

def logistic_regression(y, tx, initial_w, alpha, max_iters, threshold, method):
    
    #if method==True:
        
    """
    Do gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    w=initial_w
    losses=[]
    
    for iter in range(max_iters):
        
        # get loss and and gradient
        loss, grad = logistic_regression_aid(y, tx, w)
        
        # We print loss each 100 iteration
        if iter % 100 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
            
        # update w    
        w=w-alpha*grad  
        
        # store loss
        losses.append(loss)
        
        #Convergence criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    # visualization Pas encore non ...
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("The loss={l}".format(l=loss))

    return loss, w

def reg_logistic_regression(y, tx, initial_w, lambda_, alpha, max_iters, threshold):
    """
    Do gradient descent using reguralized logistic regression.
    Return the loss and the updated w.
    """
    w=initial_w
    losses=[]
    
    for iter in range(max_iters):
        
        # get loss and and gradient
        loss, grad = reg_logistic_regression_aid(y, tx, w, lambda_)
        
        # We print loss each 100 iteration
        if iter % 200 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
            
        # update w    
        w=w-alpha*grad  
        
        # store loss
        losses.append(loss)
        
        #Convergence criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    # visualization Pas encore non ...
    #visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent")
    print("The loss={l}".format(l=loss))

    return loss, w


def logistic_regression_aid(y, tx, w):
    """return the loss, gradient (and hessian)."""

    return calculate_logistic_loss(y, tx, w), calculate_logistic_gradient(y, tx, w)


def reg_logistic_regression_aid(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    
    # calcul new loss using the loss for simple logistic regression
    loss = calculate_logistic_loss(y, tx, w) + lambda_*(np.linalg.norm(w)**2)
    
    # calcul new gradient using the gradient for simple logistic regression
    new_grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
    
    # La Hessienne ... pas encore non ..
    # calcul à l'aide de l'ancienne hessienne
    #k = len(w)
    #new_hess = calculate_logistic_hessian(y, tx, w) + 2*lambda_*np.eye(k,k) 
    
    return loss, new_grad
    # ***************************************************
    
    
def remove_outlier_columns(big_arr):
    
    shape = big_arr.shape
    new_arr = big_arr.copy()
    for x in range(shape[0]):
        for y in range(shape[1]):
            if big_arr[x, y] == -999.:
                new_arr[x, y] = np.nan
    ind = np.isnan(new_arr).any(axis=0)
    
    for i in range(ind.shape[0]):
        if ind[i]==True:
            ind[i]=False
        else:
            ind[i]=True
            
    return new_arr[:,ind]
    

def undefToMeanMean(m):
    
    a = np.copy(m)
    a[a == -999] = float('nan')
    means = np.nanmean(a,0)
    return np.where((np.isnan(a)),means,a)

    