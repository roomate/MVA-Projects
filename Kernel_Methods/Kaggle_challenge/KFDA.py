###################################################
#########  Fisher Kernel's discriminant  ##########
###################################################
#Hugo Negrel
#April 2024

import numpy as np
import scipy.linalg
from sklearn.datasets import load_iris
import logging
from collections.abc import Callable

def split_dataset(data_x: np.array, data_y: np.array, split: int=15)->(np.array, np.array, np.array, np.array):
    dataset=np.concatenate((data_x, data_y[:,None]), axis=1)
    np.random.shuffle(dataset)
    dataset_x, dataset_y=dataset[:,:-1], dataset[:,-1]
    data_x_test, data_x_train=np.split(dataset_x, [split], axis=0)
    data_y_test, data_y_train=np.split(dataset_y, [split], axis=0)
    return data_x_train, data_y_train.astype(int), data_x_test, data_y_test.astype(int)

def generate_dataset(mu: np.array, sigma: np.array):
    mu1=mu[0]
    mu2=mu[1]
    mu3=mu[2]
    
    sigma1=sigma[0]
    sigma2=sigma[1]
    sigma3=sigma[2]
    
    X=np.concatenate((np.random.multivariate_normal([-1,mu1],np.eye(2)*sigma1, size=100),np.random.multivariate_normal([3,mu2],np.eye(2)*sigma2, size=100), np.random.multivariate_normal([1,mu3],np.eye(2)*sigma3, size=100)), axis=0)
    Y=np.array([0 for i in range(100)] + [1 for i in range(100)] + [2 for i in range(100)])
    return X,Y

def kernel(sigma: float):
    def RBF_kernel(x: np.array,y: np.array):
        return np.exp(-1/sigma*np.sum(x[:,None,:] - y[None,:,:], axis=-1)**2)
    return RBF_kernel

class Fisher_discriminant:
    def __init__(self, kernel: Callable[[np.array, np.array],np.array], nb_class: int):
        self.kernel = kernel
        self.count_class = None #number of data for each class
        self.nb_class = nb_class
        self.gram = None
        self.eigen_vec=None

    def _count_class(self,data_y: np.array):
        """
        Count the number of element in each class.
        """
        self.count_class=np.zeros(self.nb_class, dtype=int)
        for i in range(self.nb_class):
            self.count_class[i]= np.count_nonzero(data_y==i)

    def _Gram(self,data_x_train: np.array, data_x: np.array):
        """
        Compute the gram matrix given the dataset data_x_train and data_x
        """
        kern=self.kernel(data_x_train, data_x)
        return kern

    def ordering(self, data_x_train: np.array, data_y_train: np.array):
        """
        Return a list containing the order of apparition of each element in data_x in their respective class.
        The list has the following structure, for each class, we store the elements in this class and with the order in which it appears.
        This list will be usefull to compute the matrix N.
        """
        order = np.zeros((self.nb_class, len(data_x_train)), dtype=int)
        counter = np.zeros(self.nb_class)
        for i in range(len(data_x_train)):
            order[data_y_train[i]][i]=counter[data_y_train[i]]
            counter[data_y_train[i]]+=1
        del counter
        return order

    def M_matrix(self,data_x_train: np.array, data_y_train: np.array):
        """
        Compute M and N matrix. 
        To compute Mi, it is enough to compute the product of the gram matrix with the indicator of class, and then to sum over the rows. 
        Computing M from Mi and Mstar is straighforward after that.
        To compute N, it is more intricate. One needs to compute beforehand a matrix 'order' which rank the element of data_x by their order of apparition
        in their own class.
        """
        #Number of element in data_x
        n = data_x_train.shape[0]
        #Compute M_star
        M_star = np.mean(self.gram, axis = -1) #must be of size len(data_x_train)

        #Indicator matrix of class
        Indicator = np.zeros((n, self.nb_class))
        Indicator[np.arange(n), data_y_train]=1/self.count_class[data_y_train]
        #Compute Mi matrix and concatenate them into matrix M_m
        M_m = self.gram@Indicator
        #Compute M matrix
        M_matrix = np.zeros((n,n))
        for i in range(self.nb_class):
            M_matrix += np.outer(M_m[:,i] - M_star, M_m[:,i] - M_star)*self.count_class[i]
        return M_matrix

    def N_matrix(self,data_x_train: np.array, data_y_train: np.array):
        #Number of element in data_x
        n = data_x_train.shape[0]
        N_matrix = np.zeros((n,n))
        order = self.ordering(data_x_train, data_y_train) #order matrix of size nb_class x n
    
        for i in range(self.nb_class):
            l = self.count_class[i] #number of element in the class i
            B = np.zeros((n,l))
            Index=np.where(data_y_train==i)
            B[Index, order[i][Index]]=1
            Kj=self.gram@B
            N_matrix += Kj@(np.eye(l) - np.ones((l,l))*1/l)@Kj.T
        return N_matrix

    def fit(self, data_x_train: np.array, data_y_train: np.array, nb_eig_v: int):
        if (len(data_x_train) != len(data_y_train)):
            raise ValueError("Error, label and data have not same length")
        self.N = data_x_train.shape[0]
        self.gram = self._Gram(data_x_train, data_x_train)
        self._count_class(data_y_train)
        M_matrix, N_matrix = self.M_matrix(data_x_train, data_y_train), self.N_matrix(data_x_train, data_y_train)
        vecs = scipy.linalg.eig(M_matrix, N_matrix)
        self.eigen_vec=vecs[1][:,:nb_eig_v]

    def transform(self, data_x_tr: np.array, data_x_te: np.array):
        """
        Compute the projection of data_x_te onto the eigenvectors of data_x_tr
        """
        gram = self._Gram(data_x_tr,data_x_te)
        return self.eigen_vec.T@gram, gram

if __name__=="__main__":
    logger=logging.getLogger(__name__)
    logging.basicConfig(filename = 'stdout.log', level=logging.INFO)
    sigma=.01
    logger.info("Loading data...")
    logger.info(f"Selecting a RBF kernel with bandwidth {sigma}...")
    k=kernel(sigma)
    
    data_x,data_y=generate_dataset([-5,10,1], [2,1,4])
    data_x_train, data_y_train, data_x_test, data_y_test=split_dataset(data_x, data_y)
    
    Fisher=Fisher_discriminant(k, 3)
    logger.info("Fitting data...")
    Fisher.fit(data_x_train, data_y_train, 3)
    logger.info("Prediction on test dataset...")
    prediction, _=Fisher.transform(data_x_train, data_x_test)
    print(prediction.T/np.linalg.norm(prediction))