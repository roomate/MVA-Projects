import numpy as np
# KernelPCA class for dimensionality reduction and feature extraction

class KernelPCA:
    def __init__(self,kernel, r):
        self.kernel = kernel # Kernel used 
        self.alpha = None    # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.lmbda = None    # Vector of size d representing the top d eingenvalues
        self.support = None  # Data points where the features are evaluated
        self.r =r            # Number of principal components

    # Define the Gram Matrix
    def gram(self, X):
        sigma = 100
        n,d = X.shape
        Kxx = np.empty((n, n))
        for i in range(n):
            for j in range(i + 1):
                if i != j :
                  Kxx[i,j] = self.kernel(X[i,:],X[j,:],sigma)
                else :
                  Kxx[i,j] = 1
        Kxx = (Kxx + np.transpose(Kxx))/2
        return Kxx

    def compute_PCA(self, X):
        Kxx = self.gram(X)
        self.support = X
        C=np.eye(Kxx.shape[0])-np.ones(Kxx.shape)/Kxx.shape[0] # centering
        G=C.dot(Kxx).dot(C) #Centering Gram Matrix
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(G)
        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.lmbda = eigenvalues[sorted_indices[:self.r]]
        self.alpha = eigenvectors[:, sorted_indices[:self.r]] / np.sqrt(self.lmbda)

    def transform(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Compute the kernel vector between x and support
        K_x_suport = 0
        for i in range(x.shape[0]):
          res = 0
          for j in range(self.support.shape[0]):
            res += self.kernel(self.support[j] , x[i])
          K_x_support += res
        # Project the data into the kernel PCA space using the precomputed alpha
        projection = np.dot(K_x_support , self.alpha)
        # Output: vector of size N
        return projection
