import numpy as np
# Some kernel function
# We do not implement how to construct the Gram matrix here but only the basic function behind each kernel

###################################################
##############  Gaussian Kernel ###################
###################################################

def gauss_kernel(x,y,sigma=1):
      return np.exp(-((np.linalg.norm(x-y)**2)/ (2*sigma)))


###################################################
##############  Polynomial Kernel #################
###################################################


def kernel_poly(x,y,d=2):
    return (np.transpose(x) @ y )**d
