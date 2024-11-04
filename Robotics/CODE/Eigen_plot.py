# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:24:38 2023

@author: Admin
"""


from IPython import get_ipython; 
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import matplotlib.pyplot as plt

from Control_functions import Eigen_max
from scipy.spatial import ConvexHull

# NOUVEAU CODE POUR AFFICHAGE
#----------------------------
# O - INITIALISATION
# -------------------
Dt              = 0.05
N_vector_step   = 10
N_vector        = np.arange(10, 300+N_vector_step, N_vector_step)
h               = 0.8
g               = 9.81

plage_R_Q   = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0,1e1]
tmp_ref     = list(range(1, 10))
R_Q_vector  = []
for i in range(len(plage_R_Q)):
    tmp = [x * plage_R_Q[i] for x in tmp_ref]
    R_Q_vector = R_Q_vector + tmp
del tmp, i 
R_Q_vector = np.array(R_Q_vector)


Lambda_vector = Eigen_max(N_vector,R_Q_vector,Dt,h,g)


# POST-TRAITEMENT POUR AFFICHAGE
# ------------------------------
ind_R_Q_1, ind_N_1  = np.where(Lambda_vector<1)
ind_R_Q_2, ind_N_2  = np.where((Lambda_vector>1) & (Lambda_vector<2))
ind_R_Q_3, ind_N_3  = np.where((Lambda_vector>2) & (Lambda_vector<5))
ind_R_Q_5, ind_N_5  = np.where(Lambda_vector>5)

# 3 - Affichage des résultats
#----------------------------
plt.figure(num=1,figsize=(7,7))
plt.semilogy()
plt.rc('grid', linestyle="--", color='black',alpha = 0.3)
plt.grid(True)

plt.scatter(N_vector[ind_N_1], R_Q_vector[ind_R_Q_1], c='#00A86B', alpha= 1, marker='o',   label=r'       |$\lambda$| < 1')
plt.scatter(N_vector[ind_N_2], R_Q_vector[ind_R_Q_2], c='#fac205', alpha= 1, marker='o',   label=r'1 < |$\lambda$| < 2')
plt.scatter(N_vector[ind_N_3], R_Q_vector[ind_R_Q_3], c='#dc4d01', alpha= 1, marker='o',   label=r'2 < |$\lambda$| < 5')
plt.scatter(N_vector[ind_N_5], R_Q_vector[ind_R_Q_5], c='#fe0002', alpha= 1, marker='o',   label=r'       |$\lambda$| > 5')


plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# Ajouter des titres et des labels d'axes
plt.title(r'Stability criterion  |$\lambda_{\text{max}}$| for t=0.05 s')
plt.xlabel(r'Number of step (N) for control planning ')
plt.ylabel(r'Value of $Q/R$ ratio')
plt.show()




# # Généralisation du code pour faire des enveloppes convexes : 
#     # Ne marche pas très bien, les ensembles n'étant pas convexes
# #----------------------------------------------------------
# Dt_vector = [0.005, 0.01, 0.05, 0.1, 0.5]
# Dt_vector = np.array(Dt_vector)



# N_vector_step   = 10
# N_vector        = np.arange(10, 300+N_vector_step, N_vector_step)
# h               = 0.8
# g               = 9.81

# plage_R_Q   = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0,1e1]
# tmp_ref     = list(range(1, 10))
# R_Q_vector  = []
# for i in range(len(plage_R_Q)):
#     tmp = [x * plage_R_Q[i] for x in tmp_ref]
#     R_Q_vector = R_Q_vector + tmp
# del tmp, i 
# R_Q_vector = np.array(R_Q_vector)

# Lambda_vector_2 = [None]*int(np.size(Dt_vector))

# plt.figure(num=2,figsize=(7,7))
# plt.semilogy()
# plt.rc('grid', linestyle="--", color='black',alpha = 0.3)
# plt.grid(True)

# for i in range(np.size(Dt_vector)):
#     print(i)
#     Dt = Dt_vector[i]
    
#     Lambda_vector_2[i] = Eigen_max(N_vector,R_Q_vector,Dt,h,g)
    
#     ind_R_Q, ind_N  = np.where(Lambda_vector_2[i]<1)
#     tmp_R_Q         = R_Q_vector[ind_R_Q]
#     tmp_N           = N_vector[ind_N]
    
#     tmp_R_Q         = tmp_R_Q.reshape((-1,1))
#     tmp_N           = tmp_N.reshape((-1,1))
    
#     TMP             = np.concatenate((tmp_R_Q,tmp_N),axis = 1)   

#     print(np.shape(TMP))
    
#     hull = ConvexHull(TMP)
#     for simplex in hull.simplices:
#         plt.plot(TMP[simplex, 1], TMP[simplex, 0], 'k-')
#         plt.scatter(tmp_N,tmp_R_Q)
     










