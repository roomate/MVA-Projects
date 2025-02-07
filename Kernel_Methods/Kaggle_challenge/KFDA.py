###################################################
#########  Fisher Kernel's discriminant  ##########
###################################################
#Hugo Negrel
#April 2024
#The mathematical details about KFDA can be be found at: 

import numpy as np
import scipy.linalg
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin, ClassifierMixin, BaseEstimator
from sklearn.neighbors import NearestCentroid
from sklearn.utils.validation import check_is_fitted
from scipy.linalg import eigh
import logging
from collections.abc import Callable
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def split_dataset(X: np.array, y: np.array, split: int=20)->(np.array, np.array, np.array, np.array):
    X_test, X = np.split(X, [split], axis=0)
    y_test, y = np.split(y, [split], axis=0)
    return X, y.astype(int), X_test, y_test.astype(int)

def generate_dataset(mu: np.array, sigma: np.array, size: int=100):
    mu1=mu[0]
    mu2=mu[1]
    mu3=mu[2]
    
    sigma1=sigma[0]
    sigma2=sigma[1]
    sigma3=sigma[2]
    
    X=np.concatenate((np.random.multivariate_normal([4,mu1],np.eye(2)*sigma1, size=size),np.random.multivariate_normal([-2,mu2],np.eye(2)*sigma2, size=size), np.random.multivariate_normal([5,mu3],np.eye(2)*sigma3, size=size)), axis=0)
    Y=np.array([0 for i in range(size)] + [1 for i in range(size)] + [2 for i in range(size)])
    Z = np.concatenate((X, Y[:,None]), axis=1)
    np.random.shuffle(Z)
    return Z[:,:2], Z[:,2]

def symetrize(M: np.ndarray) -> None:
    M[:,:] = (M + M.T)/2

def kernel(name: str, *args) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if name is None:
        name='RBF'
    if name == 'RBF':
        if len(args) == 0:
            args=[1]
        sigma = args[0]
        def RBF_kernel(x: np.ndarray,y: np.ndarray):
            return np.exp(-1/sigma*np.sum(x[:,None,:] - y[None,:,:], axis=-1)**2)
        return RBF_kernel
    if name == 'linear':
        def linear_kernel(x: np.ndarray, y: np.ndarray):
            return x@y.T
        return linear_kernel
    
def check_pd(M: np.ndarray) -> bool:
    try:
        scipy.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return False
    return True


class KFDA(TransformerMixin, ClassifierMixin, BaseEstimator):
    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray], np.ndarray], nb_class: int, epsilon: float, nb_eigv: int):
        self.kernel = kernel
        self.count_class = None #number of data for each class
        self.nb_class = None
        self.gram = None
        self.nb_eigv = nb_eigv
        self.eigen_vec=None
        self.epsilon = epsilon

    def _count_class(self, data_y: np.array):
        """
        Count the number of element in each class.
        """
        y_unique = np.unique(data_y)
        self.count_class = np.array([np.count_nonzero(data_y == elem) for elem in y_unique]).astype('int')

    def _Gram(self,X: np.array, data_x: np.array):
        """
        Compute the gram matrix given the dataset X and data_x
        """
        kern=self.kernel(X, data_x)
        return kern

    def M_matrix(self, X: np.array, y_one_hot: np.array):

        """
        Compute M matrix. 
        """

        #Compute M_star
        M_star = np.mean(self.gram, axis = -1) #must be of size len(X)

        m_class = y_one_hot.T@self.gram
        
        M = (m_class - M_star).T@(m_class - M_star)
        return M

    def N_matrix(self, X: np.array, y_one_hot: np.ndarray):
        #Number of element in data_x
        n = X.shape[0]
        N = np.zeros((n,n))

        for i in range(self.nb_class):
            l = self.count_class[i] #number of element in the class i
            Index = np.where(y_one_hot[:,i] == 1)[0]
            Kj = self.gram[:,Index]
            N += Kj@(np.eye(l) - np.ones((l,l))*1/l)@Kj.T
        return N

    def fit(self, X: np.ndarray, y: np.array):
        if (len(X) != len(y)):
            raise ValueError("Error, label and data have not same length")
        
        self.X_ = X
        self.y_ = y
        
        self.gram = self._Gram(X, X)
        self.nb_class = len(np.unique(y))
        self._count_class(y)
        
        enc = OneHotEncoder()
        y_one_hot = enc.fit_transform(y[:,None]).toarray() #One hot encoding 

        M, N = self.M_matrix(X, y_one_hot), self.N_matrix(X, y_one_hot)
        symetrize(N)
        symetrize(M)
        
        N += np.eye(len(N))*self.epsilon #For better numerical stability 

        _, vecs = eigh(M, N, subset_by_index=[len(N) - self.nb_eigv, len(N) - 1])
        self.eigen_vec = vecs
        
        m_class = y_one_hot.T@self.gram
        
        self.centroids = m_class@vecs
        
        self.clf = NearestCentroid()
        self.clf.fit(self.centroids, np.unique(y))
        
        return self #Return the classifier

    def transform(self, X_test: np.array):
        """
        Compute the projection of data_x_te onto the eigenvectors of data_x_tr
        """
        check_is_fitted(self)
        
        gram = self._Gram(X_test, self.X_)
        return gram@self.eigen_vec
    
    def predict(self, X_test: np.ndarray):
        
        check_is_fitted(self)
        
        X_transf = self.transform(self.X_, X_test)
        
        return self.clf.predict(X_transf)
        
if __name__=="__main__":
    logger=logging.getLogger(__name__)
    logging.basicConfig(filename = 'stdout.log', level=logging.INFO)
    sigma=1
    logger.info("Loading data...")
    logger.info(f"Selecting a RBF kernel with bandwidth {sigma}...")
    k=kernel('RBF')

    data_x,data_y=generate_dataset([-20,10,1], [.1, 1,.1], size=300)
    X, y, X_test, y_test=split_dataset(data_x, data_y)

    Fisher=KFDA(k, 3, epsilon=1e-8, nb_eigv=3)
    logger.info("Fitting data...")
    Fisher=Fisher.fit(X, y)
    logger.info("Prediction on test dataset...")
    prediction=Fisher.transform(X_test)
    prediction=normalize(prediction)

    color = ('r', 'g', 'b')
    fig = plt.figure()
    for point, label in zip(X_test, y_test):
        plt.scatter(point[0], point[1], color=color[label], alpha = .5)
        plt.title("Before data separation")

    fig = plt.figure()
    for point, label in zip(prediction, y_test):
        plt.scatter(point[0], point[2], color=color[label], alpha = .5)
        plt.title("After KFDA transform")
    plt.legend()