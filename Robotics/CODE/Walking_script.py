# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 20:37:52 2023

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt

from Walking_pattern_2 import walking_pattern, foot_print, extrem_pos
from Control_functions import Control_ana


# 0 - INITIALISATION DES DONNEES
#------------------------------------------------------------------------------
# Motif de la marche
wait_duration   = 1.5       # durée de la phase d'attente (s)
dc_duration     = 0.2       # durée de la phase de double contact (s)
step_duration   = 1         # durée d'un pas du robot (s)
t_0             = 0         # instant initial (s)
Dt              = 0.01      # pas de temps de la trajectoire (s)

pos_ini         = np.reshape(np.array([0,0.2,0,-0.2]),(4,1))    # position initial de chaque pied
step_size       = 0.7       # distance parcourue en un pas (m)
N_step          = 6         # nombre de pas du motif de marche

# Géométrie des pieds
foot_length     = 0.3       # longueur du pied (m)
foot_width      = 0.1       # largeur du pied (m)

# Loi de commande
hori    = 200               # horizon de calcul de la loi de commande [nbr échantillon]
h       = 0.8               # hauteur du CdG [m]
g       = 9.81              # constante de gravité [m/s²]
ratio   = 10e-6             # ration R/Q



# 1 - CREATION DU MOTIF DE MARCHE
#------------------------------------------------------------------------------
# Création des données : motif de marche
vect_T, vect_pos, cont_G, cont_D, _  = \
    walking_pattern(wait_duration,dc_duration, t_0, Dt, pos_ini, step_duration, step_size, N_step)

# Création des données : zone de stabilité (~enveloppe convexe)
Env_XY = extrem_pos(foot_length, foot_width, vect_pos, cont_G, cont_D)

# Création des données : Z_REF (trajectoire de référence du Centre de Pression)
Z_ref       = np.ones((2,np.size(Env_XY[0,:])))
Z_ref[0,:]  = 0.5 * (Env_XY[0,:] + Env_XY[1,:])     # position de reférence selon X du Centre de Pression (m)
Z_ref[1,:]  = 0.5 * (Env_XY[2,:] + Env_XY[3,:])     # position de reférence selon Y du Centre de Pression (m)

# Création des données : Loi de commande, trajectoire réalisée du CdG et trajectoire réalisée CdP
x_0 = np.array([Z_ref[0,0],0.,0.])                    # [x_ini, v_ini, a_ini] selon l'axe X
y_0 = np.array([Z_ref[1,0],0.,0.])                    # [y_ini, v_ini, a_ini] selon l'axe y
U_x, X_hat, Z_x = Control_ana(x_0, Z_ref[0,:],\
                              vect_T, hori,h,g,ratio)
U_y, Y_hat, Z_y = Control_ana(y_0, Z_ref[1,:],\
                              vect_T, hori,h,g,ratio)
    
del x_0, y_0

# # Création des données : instants de contact pour le pied gauche et droit
# index_G = [index for index, value in enumerate(cont_G) if "G_contact" in value]
# index_D = [index for index, value in enumerate(cont_D) if "D_contact" in value]
    

# 2 - CREATION DES EMPREINTES
#------------------------------------------------------------------------------
# Création des données
Empreinte_G, Empreinte_D = foot_print(foot_length, foot_width, vect_pos, cont_G, cont_D)




# 3 - AFFICHAGE DES DONNEES : FIGURE 1 (couloir de stabilité)
#------------------------------------------------------------------------------
fig, ax     = plt.subplots(2, 1, figsize=(10, 5),sharex=True)

# Figure 1 : Axe sagittale
Left_line,  = ax[0].plot(vect_T,vect_pos[0,:],label='Center of left foot')      # our récupérer la couleur automaituqment choisie
Right_line, = ax[0].plot(vect_T,vect_pos[2,:],label='Center of right foot')      # our récupérer la couleur automaituqment choisie
ax[0].plot(vect_T,Env_XY[0,:] ,linestyle='--',color='black',label='Stability corridor')
ax[0].plot(vect_T,Env_XY[1,:],linestyle='--',color='black')

# Figure 1 : Axe transverse
ax[1].plot(vect_T,vect_pos[1,:])
ax[1].plot(vect_T,vect_pos[3,:]) 
ax[1].plot(vect_T,Env_XY[2,:] ,linestyle='--',color='black')
ax[1].plot(vect_T,Env_XY[3,:],linestyle='--',color='black')

# Figure 1 : Titres et légendes
ax[0].set_title("Evolution of the stability corridor and foot placements")
ax[0].set_ylabel('Sagittal direction X [m]')
ax[0].legend(loc = 'best')
ax[0].grid(True, linestyle='--', linewidth=1, color='black')

ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Transverse direction Y [m]')
ax[1].grid(True, linestyle='--', linewidth=1, color='black')

plt.show

# Figure 1 : Récupération des couleurs
color_left = Left_line.get_color()
color_right = Right_line.get_color()
del Left_line, Right_line


# 4 - AFFICHAGE DES DONNEES : FIGURE 2 (Z_ref et couloir de stabilité)
#------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 5),sharex=True)

# Figure 2 : Axe sagittale
ax[0].plot(vect_T,Z_ref[0,:],label='CoP : reference position',color='red')
ax[0].plot(vect_T,Env_XY[0,:] ,linestyle='--',color='black',label='Stability corridor')
ax[0].plot(vect_T,Env_XY[1,:],linestyle='--',color='black')

# Figure 2 : Axe transverse
ax[1].plot(vect_T,Z_ref[1,:],label='CoP : reference position',color='red')
ax[1].plot(vect_T,Env_XY[2,:] ,linestyle='--',color='black')
ax[1].plot(vect_T,Env_XY[3,:],linestyle='--',color='black')

# Figure 2 : Titres et légendes
ax[0].set_title("Evolution of the stability corridor and reference trajectories of CoP")
ax[0].set_ylabel('Sagittal direction X [m]')
ax[0].grid(True, linestyle='--', linewidth=1, color='black')
ax[0].legend(loc = 'best')

ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Transverse direction Y [m]')
ax[1].grid(True, linestyle='--', linewidth=1, color='black')

plt.show


# 5 - AFFICHAGE DES DONNEES : FIGURE 3 (Empreintes de pieds)
#------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 5),sharex=True)

# Figure 3 : Pied gauche
for i in range(len(Empreinte_G)): 
    if i==0:
        ax.fill(Empreinte_G[i][0,:],Empreinte_G[i][1,:],c=color_left, label ='Left foot',alpha=0.3)     # c='blue'
    else:
        ax.fill(Empreinte_G[i][0,:],Empreinte_G[i][1,:],c=color_left,alpha=0.8)                         # c='blue'

# Figure 3 : Pied droit
for i in range(len(Empreinte_D)):
    if i==0:
        ax.fill(Empreinte_D[i][0,:],Empreinte_D[i][1,:],c=color_right,label ='Right foot',alpha=0.3)    # c='#fe0002'
    else:
        ax.fill(Empreinte_D[i][0,:],Empreinte_D[i][1,:],c=color_right,alpha=0.8)                        # c='#fe0002'

# Figure 3 : Z_ref
ax.plot(Z_ref[0,:],Z_ref[1,:],label='CoP : ref.',color='red')
ax.plot(X_hat[:,0],Y_hat[:,0],label='CoM',color='green')


ax.set_title("Foot prints and trajectories of CoP and CoM")
ax.set_xlabel('Sagittal direction X [m]')
ax.set_ylabel('Transverse direction Y [m]')
ax.grid(True, linestyle='--', linewidth=1, color='black')
ax.legend(loc = 'best')

plt.show



# 6 - AFFICHAGE DES DONNEES : FIGURE 4 (Trajectoires)
#------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 5),sharex=True)

color_X = '#9a0eea'
color_Y = '#a83c09'

# Figure 2 : Tracé des lois de commande
ax[0].plot(vect_T, U_x[:-1], label='Sagittal axis', color=color_X)
ax[0].plot(vect_T, U_y[:-1], label='Transverse axis', color=color_Y)
ax[0].set_ylabel(r'Command law [m/s3]')
ax[0].grid(True, linestyle='--', linewidth=1, color='black')
ax[0].legend(loc = 'upper left')

# Figure 2 : Tracé de la trajectoire du CdG
ax[1].plot(vect_T, X_hat[:-1,0], label='Sagittal axis',color=color_X)
ax[1].set_ylabel('CoM trajectory [m]', color=color_X)
ax[1].tick_params(axis='y', labelcolor=color_X)
ax[1].grid(True, linestyle='--', linewidth=1, color='black')

ax2 = ax[1].twinx()
ax2.plot(vect_T, Y_hat[:-1,0], label='Transverse axis',color=color_Y)
ax2.set_ylabel('CoM trajectory [m]', color=color_Y)
ax2.tick_params(axis='y', labelcolor=color_Y)

lines, labels = ax[1].get_legend_handles_labels()               # Fusion des légendes
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

ax[1].set_xlabel('Time [s]')
ax[0].set_title("Evolution of control law and CoM position")
plt.show

del lines, lines2, labels, labels2




# 7 - AFFICHAGE DES DONNEES : FIGURE 2 (Z_ref et couloir de stabilité)
#------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 5),sharex=True)

color_X = '#9a0eea'
color_Y = '#a83c09'

# Figure 7 : Axe sagittale
ax[0].plot(vect_T,Z_x[1:,:],label='CoP',color=color_X)
ax[0].plot(vect_T,Env_XY[0,:] ,linestyle='--',color='black',label='Stability corridor')
ax[0].plot(vect_T,Env_XY[1,:],linestyle='--',color='black')

# Figure 7 : Axe transverse
ax[1].plot(vect_T,Z_y[1:,:],label='CoP',color=color_Y)
ax[1].plot(vect_T,Env_XY[2,:] ,linestyle='--',color='black',label='Stability corridor')
ax[1].plot(vect_T,Env_XY[3,:],linestyle='--',color='black')

# Figure 7 : Titres et légendes
ax[0].set_title("Evolution of the stability corridor and trajectories of CoP")
ax[0].set_ylabel('Sagittal direction X [m]')
ax[0].grid(True, linestyle='--', linewidth=1, color='black')
ax[0].legend(loc = 'upper left')

ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Transverse direction Y [m]')
ax[1].grid(True, linestyle='--', linewidth=1, color='black')
ax[1].legend(loc = 'lower left')

plt.show







# Fusionner les légendes


# ax2 = ax[0].twinx()
# color = 'tab:red'
# ax2.set_ylabel(r'Y Command law [m/s^3]', color=color)
# ax2.plot(vect_T, U_y[:-1], color=color)
# ax2.tick_params(axis='y', labelcolor=color)





# # Affichage des données : Figure 1
# fig, ax = plt.subplots(2, 1, figsize=(10, 5),sharex=True)
# Left_line, = ax[0].plot(vect_T,vect_pos[0,:],label='Center of left foot')
# Right_line, = ax[0].plot(vect_T,vect_pos[2,:],label='Center of right foot')
# ax[0].plot(vect_T,Env_XY[0,:] ,linestyle='--',color='black',label='Stability corridor')
# ax[0].plot(vect_T,Env_XY[1,:],linestyle='--',color='black')          # ,label='Stabilité : Xmax')
# # ax[0].plot(vect_T,Env_XY[0,:] ,linestyle='--',color='black',label='Stabilité : Xmin')

# color_left = Left_line.get_color()
# color_right = Right_line.get_color()
# del Left_line, Right_line

# ax[1].plot(vect_T,vect_pos[1,:])                                    # ,label='Pied gauche')
# ax[1].plot(vect_T,vect_pos[3,:])                                    # ,label='Pied droit')
# ax[1].plot(vect_T,Env_XY[2,:] ,linestyle='--',color='black')        # ,label='Stabilité : Ymin')
# ax[1].plot(vect_T,Env_XY[3,:],linestyle='--',color='black')          # ,label='Stabilité : Ymax')

# ax[0].set_title("Evolution of the stability corridor and foot placements")
# ax[0].set_ylabel('Sagittal direction X [m]')
# ax[0].legend(loc = 'best')
# ax[0].grid(True, linestyle='--', linewidth=1, color='black')

# ax[1].set_xlabel('Durée [s]')
# ax[1].set_ylabel('Transverse direction Y [m]')
# ax[1].grid(True, linestyle='--', linewidth=1, color='black')

# plt.show



# # Affichage des données : Figure 2
# fig, ax = plt.subplots(2, 1, figsize=(10, 5),sharex=True)
# ax[0].plot(vect_T,Z_ref[0,:],label='CoP : reference position',color='red')
# ax[0].plot(vect_T,Env_XY[0,:] ,linestyle=':',color='black',label='Stability corridor')
# ax[0].plot(vect_T,Env_XY[1,:],linestyle=':',color='black')

# ax[1].plot(vect_T,Z_ref[1,:],label='CoP : reference position',color='red')
# ax[1].plot(vect_T,Env_XY[2,:] ,linestyle='--',color='black')
# ax[1].plot(vect_T,Env_XY[3,:],linestyle=':',color='black')

# ax[0].set_title("Evolution of the stability corridor and reference CoP")
# ax[0].set_ylabel('Sagittal direction X [m]')
# ax[0].grid(True, linestyle='--', linewidth=1, color='black')
# ax[0].legend(loc = 'best')

# ax[1].set_xlabel('Durée [s]')
# ax[1].set_ylabel('Transverse direction Y [m]')
# ax[1].grid(True, linestyle='--', linewidth=1, color='black')

# plt.show





    
# # 2 - CREATION DES EMPREINTES
# #---------------------------------------------------------
# # # Création des données
# # Empreinte_G, Empreinte_D = foot_print(foot_length, foot_width, vect_pos, cont_G, cont_D)

# # Affichage des données
# fig, ax = plt.subplots(1, 1, figsize=(10, 5),sharex=True)
# for i in range(len(Empreinte_G)): 
#     if i==0:
#         ax.fill(Empreinte_G[i][0,:],Empreinte_G[i][1,:],c=color_left, label ='Left foot',alpha=0.3)     # c='blue'
#     else:
#         ax.fill(Empreinte_G[i][0,:],Empreinte_G[i][1,:],c=color_left,alpha=0.8)                         # c='blue'
# for i in range(len(Empreinte_D)):
#     if i==0:
#         ax.fill(Empreinte_D[i][0,:],Empreinte_D[i][1,:],c=color_right,label ='Right foot',alpha=0.3)    # c='#fe0002'
#     else:
#         ax.fill(Empreinte_D[i][0,:],Empreinte_D[i][1,:],c=color_right,alpha=0.8)                        # c='#fe0002'

# ax.plot(Z_ref[0,:],Z_ref[1,:],label='CoP : ref.',color='red')

# ax.set_title("Foot prints and trajectories of CoP and CoM")
# ax.set_xlabel('Sagittal direction X [m]')
# ax.set_ylabel('Transverse direction Y [m]')
# ax.grid(True, linestyle='--', linewidth=1, color='black')
# ax.legend(loc = 'best')

# plt.show