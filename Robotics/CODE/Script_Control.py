# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:47:58 2023

@author: Admin
"""


# -----------------------------------------------------------------------------
# XX - ZONE DE SCRIPT POUR TEST DES FONCTIONS DE "Control_functions.py"
#------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

from Control_functions import Construction_matrix, Compute_A_B, Compute_State, Compute_command_ana, Control_ana



# 00 - INITIALISATION DES PARAMETRES
#-----------------------------------
D_simu   = 10      # Durée de la simulation [s]
Dt       = 0.005   # Intervalle entre échantillon [s]
n_simu   = int(D_simu/Dt)
Vect_tps = np.linspace(0,D_simu,n_simu+1)

hori    = 300  # horizon de calcul de la loi de commande [nbr échantillon]

h       = 0.8       # hauteur du CdG
g       = 9.81      # constante de gravité [m/s²]
ratio   = 10e-6    # ration R/G

x_0     = np.zeros((3,1))   # Vecteur d'état à l'instant initial

# 0 - Création du vecteur Z_ref clear
#------------------------------
# fonction sinus
Z_ampl      = 0.3
Z_freq      = 1  # Hz
Z_R_sin     = Z_ampl * np.sin(2 * np.pi * Z_freq * Vect_tps)

# 0 - Création du vecteur Z_ref : fonctions échelons
n_step      = 4
Z_R_ech     = [0]*int(0.5*1/Dt) \
            + ([-Z_ampl]*int(0.5*1/Dt) + [Z_ampl]*int(0.5*1/Dt))*n_step \
            + [0]*int(0.5*1/Dt) \
            + [0]
Vect_tps_ech= list(range(0, np.size(Z_R_ech), 1))
Vect_tps_ech = np.array(Vect_tps_ech)*Dt
Vect_tps_ech = np.reshape(Vect_tps_ech,(-1,1))


# 1 - Loi de commande et affichage : Z_REF sinusoïdale
#------------------------------------------------------------
# U_sin, X_hat_sin, Z_sin = Control_ana(x_0, Z_R_sin, Vect_tps,hori,h,g,ratio)
# fig, ax = plt.subplots(4, 1, figsize=(10, 5))
# ax[0].plot(Vect_tps,Z_R_sin,label='Z_REF')
# ax[0].legend(loc = 'best')

# ax[1].plot(Z_sin[1:,:],label='Z_est')
# ax[1].legend(loc = 'best')

# ax[2].plot(U_sin,label='Commande analytique')
# ax[2].legend(loc = 'best')

# ax[3].plot(X_hat_sin[:,0],label='Position CdG')
# ax[3].legend(loc = 'best')

# plt.show

# 1 - Loi de commande et affichage : Z_REF échelon
#------------------------------------------------------------
U_ech, X_hat_ech, Z_ech = Control_ana(x_0, Z_R_ech, Vect_tps_ech,hori,h,g,ratio)
fig, ax = plt.subplots(4, 1, figsize=(10, 5))
ax[0].plot(Vect_tps_ech,Z_R_ech,label='Z_REF')
ax[0].legend(loc = 'best')

ax[1].plot(Z_ech[1:,:],label='Z_est')
ax[1].legend(loc = 'best')

ax[2].plot(U_ech,label='Commande analytique')
ax[2].legend(loc = 'best')

ax[3].plot(X_hat_ech[:,0],label='Position CdG')
ax[3].legend(loc = 'best')

plt.show