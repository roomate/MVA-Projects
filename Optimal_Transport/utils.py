import numpy as np
from collections.abc import Callable
import time
###
def find(Y):
    """
    Returns the index sought for finding the root 

    Args : 
        Y : vector we want to project on the unitary simplex (list n x 1)
    """
    for i in range(1,len(Y)): #We start from the largest element
        if np.sum([Y[j] - Y[i] for j in range(i)]) - 1 >= 0:
            return i
    return -1

def dist(x,y):
    """
    return euclidian distance between x and y, with a more stable implementation
    Outputs : 
        ||x - y||^2
    """
    return np.dot(x,x) - 2*np.dot(x,y) + np.dot(y,y)

def root(Y):
    """
    Returns the root of h, i.e the optimal Lagrange multiplier of kkt conditions
    """
    Y_p = np.sort(Y) #We sort elements from smaller to larger
    Y_p = np.flip(Y_p)
    i = find(Y_p)
    if i >= 0:
        return (np.sum(Y_p[:i]) - 1)/i
    else:
        return (np.sum(Y) - 1)/len(Y)

def entropy(Y, gamma):
    return np.sum(Y*np.log(Y))*gamma

def proj_simplexe(Y): 
    mu = root(Y)
    if mu == 'non feasible':
        return Y
    else:
        return np.maximum(Y - mu, np.zeros(len(Y)))

def conjug_entropy(x, gamma):
    return gamma*np.sum(np.exp(x/gamma - 1))

def grad_conjug_entropy(x, gamma):
    return np.exp(x/gamma - 1)

def max_entropy(x,gamma,b):
    c = x.max()/gamma #Log-sum-exp tricks
    return gamma*np.log(np.sum(np.exp(x/gamma - c))) - np.log(b)*gamma + c*gamma

def grad_max_entropy(x, gamma):
    L = np.exp(x/gamma)
    return L/np.sum(L)

def squared_norm(x,gamma):
    return np.linalg.norm(x)**2 * gamma/2

def conjug_squared_norm(x,gamma):
    return 1/(2*gamma)*np.sum([x[i]**2 for i in range(len(x)) if x[i] > 0])

def grad_conjug_squared_norm(x,gamma):
    return np.maximum(x,0)/gamma

def grad_max_squared_norm(x,gamma,b):
    return proj_simplexe(x/(gamma*b))

def max_squared_norm(x,gamma,b):
    y = grad_max_squared_norm(x, gamma, b)
    return np.dot(x,y) - gamma*b/2*np.dot(y,y)

def mean_vector(T: np.ndarray, y: np.ndarray):
    """
    new color after transport of y with T
    Args:
        T : transport plan (list n x m)
        y : cluster of the source (or reference) image
    Outputs : 
        solution of the barycentric problem projection
    """
    return T@y/np.sum(T, axis = 1)[:,None]

def mean_tensor(T: np.ndarray, y: int, n_src: int, n_ref: int):
    """
    new color after transport of y with T
    Args:
        T : transport plan (list n x m)
        y : cluster of the source (or reference) image (int)
        n_src : number of cluster of the source image (int)
        n_ref : number of cluster of the reference image (int)
    Outputs : 
        solution of the barycentric problem projection
    """
    x = T@y
    x /= np.sum(T, axis = 1)
    return x

#Wrapper to measure the time elapsed during computation
def clock(f: Callable):
    def measure_time(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        print("Total time spent is:", end - start)
        return res
    return measure_time
