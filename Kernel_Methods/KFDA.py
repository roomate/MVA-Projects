###################################################
#########  Fisher Kernel's discriminant  ##########
###################################################

class Fisher_discriminant:
    def __init__(self, kernel, nb_class = 10):
        self.kernel = kernel
        self.count_class = None #number of data for each class
        self.n_class = nb_class
        self.gram = None

    def _count_class(self,data_y):
        """
        Count the number of element in each class.
        """
        self.count_class = np.zeros(data_y.shape)
        for i in data_y:
            self.count_class[i - 1] += 1

    def _Gram(self,data_x,data_x_p):
        """
        Compute the gram matrix given the dataset data_x and data_x_p
        """
        kern = np.empty((data_x.shape[0], data_x_p.shape[0]))
        if (data_x_p.shape[0] == data_x.shape[0]):
            for i in range(data_x.shape[0]):
                for j in range(i + 1):
                    kern[i,j] = self.kernel(data_x[i],data_x_p[j])
                    kern[j, i] = kern[i, j]
        else:
            for i in range(data_x.shape[0]):
                for j in range(data_x_p.shape[0]):
                    kern[i,j] = self.kernel(data_x[i], data_x_p[j])
        return kern

    def ordering(self,data_x, data_y):
        """
        Return a list containing the order of apparition of each element in data_x in their respective class.
        The list has the following structure, for each class, we store the elements in this class and with the order in which it appears.
        This list will be usefull to compute the matrix N.
        """
        order = np.zeros((self.n_class,len(data_x)))
        counter = np.zeros(self.n_class)
        for i in range(len(data_x)):
            order[data_y[i] - 1][i] = counter[data_y[i] - 1]
            counter[data_y[i] - 1] += 1
        return order

    def M_matrix(self,data_x, data_y):
        """
        Compute M and N matrix. 
        To compute Mi, it is enough to compute the product of the gram matrix with the indicator of class, and then to sum over the rows. 
        Computing M from Mi and Mstar is straighforward after that.
        To compute N, it is more intricate. One needs to compute beforehand a matrix 'order' which rank the element of data_x by their order of apparition
        in their own class
        """
        #Number of element in data_x
        n = data_x.shape[0]
        #Compute M_star
        M_star = np.sum(self.gram, axis = 1)/n #must be of size len(data_x)

        #Indicator matrix of class
        Indicator = np.zeros((n, self.n_class))
        for i in range(len(data_x)):
            Indicator[i,int(data_y[i] - 1)] = 1/self.count_class[int(data_y[i] - 1)]
        #Compute Mi matrix and concatenate them into matrix M_m
        M_m = np.dot(self.gram,Indicator)
        #Compute M matrix
        M_matri = np.zeros((n,n))
        for i in range(self.n_class):
            M_matri += np.outer(M_m[:,i] - M_star, M_m[:,i] - M_star)*self.count_class[i]
        return M_matri
        
    def N_matrix(self,data_x,data_y):
        #Number of element in data_x
        n = data_x.shape[0]

        N_matrix = np.zeros((n,n))
        order = self.ordering(data_x, data_y) #order matrix of size n_class x n
        for i in range(self.n_class):
            l = int(self.count_class[i]) #number of element in the class i
            B = np.zeros((n,l))
            for j in range(len(data_x)):
                if data_y[j] - 1 == i:
                    B[j,int(order[i][j])] = 1 #Place a 1 if the jth element of data_x belong to class i. The column depends on the order of apparition in the class j.
            Kj = np.dot(self.gram,B)
            N_matrix += Kj.dot(np.eye(l) - np.ones((l,l))*1/l).dot(Kj.T)
        return N_matrix

    def fit(self, data_x, data_y):
        if (len(data_x) != len(data_y)):
            print("error, label and data have not same length")
            return 0
        self.N = data_x.shape[0]
        self.gram = self._Gram(data_x, data_x)
        self._count_class(data_y)
        M_matri, N_matrix = self.M_matrix(data_x, data_y)
        vecs = scipy.linalg.eig(M_matri, N_matrix)
        return vecs[1][:,:self.n_class - 1]

    def transform(self, eig_vec, data_x_tr, data_x_te):
        """
        Compute the projection of data_x onto eigenvectors.
        """
        gram = self._Gram(data_x_tr,data_x_te)
        return np.dot(eig_vec,gram), gram
