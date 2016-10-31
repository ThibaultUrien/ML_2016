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
    
    # Define initial w
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute gradient and loss
        G=compute_gradient(y,tx,w)
        loss=compute_loss_mse(y,tx, w)
        # update w by gradient
        w=w-gamma*G

        
        # print loss every 100 iterations
        if n_iter % 100 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))
            
        
    print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return w

def least_square_SGD(y, tx, batch_size, initial_w, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    
    # Define initial w 
    w = initial_w;
    
    for n_iter in range(max_epochs):

        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w-gamma*grad
            loss=compute_loss_mse(y,tx,w)
          
        if n_iter % 10 == 0:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_epochs - 1, l=loss))
      
    return w

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
    l=1
    if(len(tx.shape) > 1):
        l = tx.shape[1]
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
    #return np.sum(loss)
    logistic_loss = np.log (1+np.exp(np.dot(tx,w)))-y*np.dot(tx,w)
    return np.sum(logistic_loss)

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""

    err= sigmoid(np.dot(tx,w))-y
    return np.dot(tx.T,err)

def calculate_logistic_hessian(tx, w):
    """return the hessian of the loss function."""

    sig=sigmoid(np.dot(tx,w))*(1-sigmoid(np.dot(tx,w)))

    H = np.dot(tx.T, sig*tx)

    return H

def logistic_regression(y, tx, initial_w, alpha, max_iters, method):
    
    # if method == True, we implement Gradient descent
    # if method == False, we implement Newton method   
    
    """
    Do gradient descent / Newton method using logistic regression.
    Return the loss and the updated w.
    """
    
    w=initial_w
    losses=[]
    
    for iter in range(max_iters):
        
        # get loss and and gradient
        loss, grad, hess = logistic_regression_aid(y, tx, w)
        
        # We print loss each 200 iteration
        if iter % 200 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
            
        # update w 
        
        if method == True:
            w=w-alpha*grad  
            
        else:
            w=w-alpha*np.linalg.solve(hess,grad)
    
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
    """return the loss, gradient and hessian !"""

    return calculate_logistic_loss(y, tx, w), calculate_logistic_gradient(y, tx, w), calculate_logistic_hessian(tx, w)


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


def method(i, tX):
    if i==1:
        return standardize(remove_outlier_columns(tX), mean_x=None, std_x=None)
    if i==2:
        return standardize(undefToMeanMean(tX), mean_x=None, std_x=None)
    if i==3:
        return standardize(build_poly(tX, 2), mean_x=None, std_x=None)
        


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """Nothing implemented yet just return x, which is right if deg == 1"""
    
    if(degree != 1):
        new_x = undefToMeanMean(x)
        ret=np.zeros((new_x.shape[0],new_x.shape[1]*degree))
        
        for i in np.arange(new_x.shape[1]):
            for deg in np.arange(degree):
                ret[:,2*(i)] = new_x[:,i]
                ret[:,2*(i)+1] = new_x[:,i]**(deg+1)
        return ret

    #ret=np.zeros((len(x),degree+1))
    
    #for i in np.arange(degree+1):
     #   ret[:,i]=x**(i)  
    return x
    # ***************************************************
    
    """cross validation as implemented in labs 4"""
def cross_validation(y, x, initial_w, alpha, max_iter, threshold, k_indices, k, lambda_, degree):
    """ return the loss for test and train sets """
    train_err=[]
    test_err=[]
    
    for i in k_indices:
        #print(i)
        # as x and y can have more than 1 dimetion, add  0 as direction perameter to delete lines.
        x_train, y_train = np.delete(x,i,0), np.delete(y,i,0)
        x_test, y_test = x[i], y[i]
        #print('x_train.shape = ', x_train.shape,'y_train.shape = ', y_train.shape)
        #print('x_test.shape = ', x_test.shape,'y_test.shape = ', y_test.shape)
        phi_test = build_poly(x_test, degree)
        phi_train = build_poly(x_train, degree)
        
        #w_opt, mse_tr = ridge_regression(y_train, phi_train, lambda_)
        mse_tr, w_opt = reg_logistic_regression(y_train, phi_train, initial_w, lambda_, alpha, max_iter, threshold)
        #print(mse_tr.shape, w_opt.shape)
        
        # Pour ridge regression
        #mse_te = compute_loss_mse(y_test, phi_test, w_opt)
        
        # Pour logistique penalized regression
        mse_te = calculate_logistic_loss(y_test, phi_test, w_opt) + lambda_*(np.linalg.norm(w_opt)**2)
        
        rmse_tr, rmse_te= (2*mse_tr)**(0.5), (2*mse_te)**(0.5)
        #print('mse_tr = ', mse_tr)
        #print('mse_te = ', mse_te)
        

        # On utilise la mse
        train_err.append(mse_tr)
        test_err.append(mse_te)
        
        # On utilise la rmse
        #train_err.append(rmse_tr)
        #test_err.append(rmse_te)
        
    loss_tr = np.mean(train_err)
    #print('moyenne : rmse_tr', loss_tr)
    loss_te = np.mean(test_err)
    #print('moyenne : rmse_te', loss_te)
    return loss_tr, loss_te, test_err

    
