import numpy as np
import scipy.optimize as optimize

# SVM class  for multi-class classification using the one-vs-all (OvA) strategy 
class SVM:
    def __init__(self , kernel, epsilon , C):
        self.C = C                   # Regularization parameter
        self.alpha = None            # Lagrange multipliers
        self.kernel = kernel         # Kernel used
        self.epsilon = epsilon       # Threshold to enforce sparsity 
        self.support = None          # Training data
        self.y = None                # Label of training data

    # Function that builds the Gram Matrix
    def Gram(self,X):
        N,d = X.shape
        Kxx = np.empty((N, N))
        for i in range(N):
            for j in range(i + 1):
                if i != j :
                  Kxx[i,j] = self.kernel(X[i,:],X[j,:])
                  Kxx[j, i] = Kxx[i, j]
                else :
                  Kxx[i,j] = 1
        return Kxx

    def train(self, X, y):
        self.support = X
        self.y = y
        N = len(y)
        # Define the Gram Matrix
        Gram = self.Gram(X)
        alpha_sol = np.empty((10,N))
        # Define constraints
        A = np.vstack((-np.eye(N), np.eye(N)))
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))
        for i in range(10):
          print("iteration: ", i+1)
          # We take the index of all images of class i and create a new vector
          # such that all images of label i have now a label 1 and others (-1)
          index = np.where(y == i)[0]
          y_new = - np.ones(N)
          y_new[index] = 1
          # Lagrange dual problem associated to SVM minimization
          def loss(alpha):
              return np.transpose(alpha) @ Gram @ alpha - 2 * np.transpose(alpha) @ y_new
          def grad_loss(alpha):
              return Gram @ alpha - y_new
          # Constraints on alpha of the shape :
          # -  d - C*alpha  = 0
          # -  b - A*alpha >= 0
          fun_ineq = lambda alpha:  b - np.dot(A, alpha)
          jac_ineq = lambda alpha:  - A
          constraints = ({'type': 'ineq',
                        'fun': fun_ineq ,
                        'jac': jac_ineq})
          optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints)
          alpha_sol[i] = np.where(np.abs(optRes.x) < self.epsilon, 0 , optRes.x)
          # Ensure the sparsity of the solution
          print("nb_zeros =", len(alpha_sol[i]) - np.count_nonzero(alpha_sol[i]))
        self.alpha = alpha_sol
    def predict(self, x):
        X_train = self.support
        # N is the number of samples
        N , n_features = X_train.shape
        n_pred = x.shape[0]
        # Initialize prediction
        pred = np.zeros(n_pred)
        # Compute for each predictions...
        for j in range(n_pred) :
          pred_class = np.zeros(10)
          #...compute the prediction to be in class i...
          for i in range(10):
              res = 0
              #...by computing sum_k^N [alpha_i(k) * Kernel(X_k,x_j)]
              for k in range(N) :
                  res += self.alpha[i, k] * self.kernel(X_train[k , :] , x[j , :])
              pred_class[i] = res
          # Take the index of the maximum value : it corresponds to the
          # predicted class
          pred[j] = np.argmax(pred_class)
          if j%100 ==0 :
            print(j+1,"th prediction done")
        return pred
