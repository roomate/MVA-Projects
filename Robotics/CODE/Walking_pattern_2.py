# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:42:19 2023

@author: Admin
"""

# from IPython import get_ipython; 
# get_ipython().run_line_magic('reset', '-sf')

import numpy as np

"""
Pour toutes les fonctions, on adopte le formalisme suivant : 
    
Paramètres
    duration: scalaire
                Durée de la phase [s]
    t_0     : scalaire
                Instant initial du vecteur temps [s]
    Dt      : scalaire
                Pas de temps du vecteur temps [s]
    pos_ini : [4x1] numpy array
                pos_ini[0,0]: pos. X du centre du pied Gauche [m]
                pos_ini[1,0]: pos. Y du centre du pied Gauche [m]
                pos_ini[2,0]: pos. X du centre du pied Droit [m]
                pos_ini[3,0]: pos. Z du centre du pied Droit [m]
    
Returns
    vect_T  : [n] numpy array
                Vecteur position (1D) commençant à "0" et finissant à "duration + Dt" par pas de "Dt" [s]
                On garde un pas de temps en plus pour faciliter l'extraction des futures positions initiales.
    vect_pos: [4xn]
                vect_pos[0,i]: pos. X du centre du pied Gauche au ième pas de temps du vecteur temps [m]
                vect_pos[1,i]: pos. Y du centre du pied Gauche au ième pas de temps du vecteur temps [m]
                vect_pos[2,i]: pos. X du centre du pied Droit au ième pas de temps du vecteur temps [m]
                vect_pos[3,i]: pos. Z du centre du pied Droit au ième pas de temps du vecteur temps [m]
    cont_G : [n] list
                contact[i] : indicateur sur le fait que le pied Gauche est en contact ou non avec le sol
    cont_D : [n] list
                contact[i] : indicateur sur le fait que le pied Droit est en contact ou non avec le sol
    conv_hul: [n] liste de numpy array
                conv_hul[i] : ensemble des coordonnées (X,Y) des points définissant l'enveloppe convexe 
                              au ième pas du vecteur temps [m,m]. Ces coordonnées sont stockées sous forme 
                              de taleau numpy dont la dimension est [2xK]:
                                  [0,k] : coordonnée X du kième point constituant l'enveloppe convexe à l'instant i
                                  [1,k] : coordonnée Y du kième point constituant l'enveloppe convexe à l'instant i

"""


def standing_phase(duration,t_0,Dt,pos_ini):
    """
    Création des données pour une phase de double contact.
    Paramètres
        (duration, t_0, Dt, pos_ini) : cf. commentaire en début de code
    Returns
        (vect_T, vect_pos, cont_G, cont_D, conv_hul) : cf. commentaire en début de code
    """
    
    # 0 - Initialisation
    vect_T      = np.arange(t_0,t_0+duration+Dt,Dt)
    pos_ini     = np.reshape(pos_ini,(4,1))
    pos_ini     = pos_ini.astype(float)             # Pour éviter de passe 4 heures (littéralement) à débuger
    
    # 1 - Création des sorties
    vect_pos    = np.tile(pos_ini, np.size(vect_T))      # Pour répéter le vecteur
    cont_G      = ["G_contact"]*np.size(vect_T)
    cont_D      = ["D_contact"]*np.size(vect_T)
    
    # 2 - Création de l'enveloppe convexe
    conv_hul = ["en cours de création"]
    
    
    del duration,t_0,Dt,pos_ini
    return vect_T, vect_pos, cont_G, cont_D, conv_hul



def walking_phase(duration,t_0,Dt,pos_ini,step_size,foot):
    """
    Création des données pour une phase constituée d'un unique pas, pouvant débuter
    par le pied gauche ou le pied droit. La phase réalise un pas dans la direction sagittale,
    ici choisie comme la direction X.
    Le dernier instant de la phase est une phase de double contact (convention).
    
    Paramètres
        (duration, t_0, Dt, pos_ini) : cf. commentaire en début de code
        step_size : scalaire
                Longueur d'un pas dans la direction sagittale, i.e. x [m]
        foot    : chaine de caractère
                    "gauche" : le pied gauche avance, i.e. ne touche pas le sol
                    "droit"  : le pied droit avance, i.e. ne touche pas le sol
        
    Returns
        (vect_T, vect_pos, cont_G, cont_D, conv_hul) : cf. commentaire en début de code
    """
    
    # 0 - Initialisation
    vect_T, vect_pos, cont_G, cont_D, _ = standing_phase(duration,t_0,Dt,pos_ini)
      
    # 1 - Cas d'une avancée du pied gauche
    if foot == "gauche":
        cont_G[0:-1] = ["G_non_contact"]*(np.size(vect_T)-1)    # Tous les instants en "non contact" sauf le dernier car issu de "standing phase"
        vect_pos[0,:]  = vect_pos[0,0] \
                        + np.linspace(0, step_size,num=np.size(vect_T),endpoint=True) # Interpolation linéaire de la position
    
    # 2 - Cas d'une avancée du pied droit                    
    if foot == "droit":
        cont_D[0:-1] = ["D_non_contact"]*(np.size(vect_T)-1)    # Tous les instants en "non contact" sauf le dernier car issu de "standing phase"
        vect_pos[2,:]  = vect_pos[2,0] \
                        + np.linspace(0, step_size,num=np.size(vect_T),endpoint=True) # Interpolation linéaire de la position
                        
    if (foot !="droit") & (foot!="gauche"):
        vect_T      = ["NOK"] 
        vect_pos    = ["NOK"] 
        cont_G      = ["NOK"] 
        cont_D      = ["NOK"] 
        conv_hul    = ["NOK"] 
    
    # 3 - Création de l'enveloppe convexe
    conv_hul = ["en cours de création"]
    
    del duration,t_0,Dt,pos_ini,step_size,foot
    return vect_T, vect_pos, cont_G, cont_D, conv_hul
















def walking_pattern(wait_duration,dc_duration, t_0, Dt, pos_ini, step_duration, step_size, N_step):
    """
    Création d'un motif de marche
    
    Paramètres
        wait_duration   : scalaire
                    Durée de la phase d'attente [s]
        dc_duration : scalaire
                    Durée de la phase de marche (durée d'un pas) [s]
        d_contact_duration   : scalaire
                    Durée de la phase de double contact entre deux pas [s]
        Dt            : scalaire
                    Pas de temps utilisé pour la création du vecteur temps [s]
        Step_length : scalaire
                    Longueur d'un pas dans la direction sagittale, i.e. x pour nous [m]
        N_step : scalaire
                   Nombre de pas à considérer pour la séquence [s.u.]
        Pos_init : [4x1] numpy array
                    Matrice de vecteur donnant la position initiale du pied gauche et du pied droit.
                    Pos_init[0,0]: pos. X du pied gauche [m]
                    Pos_init[1,0]: pos. Y du pied gauche [m]
                    Pos_init[2,0]: pos. X du pied droit [m]
                    Pos_init[3,0]: pos. Z du pied droit [m]
    """
    
    # 1 - Phase d'attente
    #--------------------  
    vect_T, vect_pos, cont_G, cont_D, conv_hul = \
        standing_phase(wait_duration,t_0,Dt,pos_ini)
    
    # 2 - Demi-pas (pied droit)
    #--------------------------
    # Création des données
    pos_ini = np.reshape(vect_pos[:,-1],(4,1))
    t_0     = vect_T[-1]
    foot    = "droit" 
    vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1 = \
        walking_phase(step_duration/2,t_0,Dt,pos_ini,step_size/2,foot)
    
    # Concaténation des données (suppression des dernières valeurs des tableaux initiaux)  
    vect_T      = np.concatenate((vect_T[:-1],vect_T1))
    vect_pos    = np.concatenate((vect_pos[:,0:-1],vect_pos1),axis=1)
    cont_G      = cont_G[:-1] + cont_G1
    cont_D      = cont_D[:-1] + cont_D1
    
    del vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1
    
    
    
    # 3 - Phase de double contact
    #----------------------------
    pos_ini = np.reshape(vect_pos[:,-1],(4,1))
    t_0     = vect_T[-1]
    vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1 = \
        standing_phase(dc_duration,t_0,Dt,pos_ini)
    
    # Concaténation des données (suppression des dernières valeurs des tableaux initiaux)
    vect_T      = np.concatenate((vect_T[:-1],vect_T1))
    vect_pos    = np.concatenate((vect_pos[:,0:-1],vect_pos1),axis=1)
    cont_G      = cont_G[:-1] + cont_G1
    cont_D      = cont_D[:-1] + cont_D1
    
    del vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1
    
    #--------------------------
    #   ENCHAINEMENT DES PAS
    #--------------------------
    for i in range(N_step):
        # 4 - Réalisation du pas
        #-----------------------
        if foot=="droit":
            foot = "gauche"
        else:
            if foot=="gauche":
                foot = "droit"
         
        pos_ini = np.reshape(vect_pos[:,-1],(4,1))
        t_0     = vect_T[-1]
        vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1 = \
            walking_phase(step_duration,t_0,Dt,pos_ini,step_size,foot)
            
        # Concaténation des données (suppression des dernières valeurs des tableaux initiaux)
        vect_T      = np.concatenate((vect_T[:-1],vect_T1))
        vect_pos    = np.concatenate((vect_pos[:,0:-1],vect_pos1),axis=1)
        cont_G      = cont_G[:-1] + cont_G1
        cont_D      = cont_D[:-1] + cont_D1
        
        del vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1
        
        # 5 - Phase de double contact
        #----------------------------
        pos_ini = np.reshape(vect_pos[:,-1],(4,1))
        t_0     = vect_T[-1]
        vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1 = \
            standing_phase(dc_duration,t_0,Dt,pos_ini)
        
        # Concaténation des données (suppression des dernières valeurs des tableaux initiaux)
        vect_T      = np.concatenate((vect_T[:-1],vect_T1))
        vect_pos    = np.concatenate((vect_pos[:,0:-1],vect_pos1),axis=1)
        cont_G      = cont_G[:-1] + cont_G1
        cont_D      = cont_D[:-1] + cont_D1
        
        del vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1
        
        
    # 6 - Demi-pas 
    #--------------------------
    # Création des données
    pos_ini = np.reshape(vect_pos[:,-1],(4,1))
    t_0     = vect_T[-1]
    if foot=="droit":
        foot = "gauche"
    else:
        if foot=="gauche":
            foot = "droit" 
    vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1 = \
        walking_phase(step_duration/2,t_0,Dt,pos_ini,step_size/2,foot)
    
    # Concaténation des données (suppression des dernières valeurs des tableaux initiaux)
    vect_T      = np.concatenate((vect_T[:-1],vect_T1))
    vect_pos    = np.concatenate((vect_pos[:,0:-1],vect_pos1),axis=1)
    cont_G      = cont_G[:-1] + cont_G1
    cont_D      = cont_D[:-1] + cont_D1
    
    del vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1
        
    # 7 - Phase d'attente
    #--------------------------
    # Création des données
    pos_ini = np.reshape(vect_pos[:,-1],(4,1))
    t_0     = vect_T[-1]
    vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1 = \
        standing_phase(wait_duration,t_0,Dt,pos_ini)
        
    # Concaténation des données (suppression des dernières valeurs des tableaux initiaux)
    vect_T      = np.concatenate((vect_T[:-1],vect_T1))
    vect_pos    = np.concatenate((vect_pos[:,0:-1],vect_pos1),axis=1)
    cont_G      = cont_G[:-1] + cont_G1
    cont_D      = cont_D[:-1] + cont_D1
    
    del vect_T1, vect_pos1, cont_G1, cont_D1, conv_hul1
        
    return vect_T, vect_pos, cont_G, cont_D, conv_hul    
   












def foot_print(foot_length, foot_width, vect_pos, cont_G, cont_D):
    """
    Création des coordonnées permettant de tracer l'empreinte des pas du robot.
    
    Parameters
        foot_length : scalar 
                    Longueur du pied (selon l'axe sagittal X) [m]
        foot_width : scalar 
                    Largeur du pied (selon l'axe transverse Y) [m]
        (vect_pos, cont_G, cont_D)   : cf. commentaire en début de code   

    Returns
        G_foot_list : list of [2x4] array 
                    Tableau contenant les positions au sol du pied gauche. 
                    Chaque élément de la liste renvoie une position unique du pied.
                    Chaque position est repérée par les 4 coordonnées (X,Y) des 
                    somments du pied considér comme un rectangle. Ces position sont
                    stockée dans un tableau dont la première ligne contient les
                    positions en X et la seconde les positions en Y.
        D_foot_list : list of [2x4] array 
                    Tableau contenant les positions au sol du pied droit.
                    La construction de la variable est identique à celle <G_foot_list>
    """

    # 1 - Création des tableaux d'indices pour savoir s'il y a contact ou non
    tmp = 'G_contact'
    G_ind_cont      = [i for i, chaine in enumerate(cont_G) if tmp in chaine]
    tmp = 'D_contact'
    D_ind_cont      = [i for i, chaine in enumerate(cont_D) if tmp in chaine]

    G_ind_cont      = np.array(G_ind_cont)
    D_ind_cont      = np.array(D_ind_cont)
    del tmp, cont_G, cont_D
    
    # 2 - Création des empreintes de pas
    G_unique = np.unique(vect_pos[0:2,G_ind_cont], axis=1)
    D_unique = np.unique(vect_pos[2:,D_ind_cont], axis=1)
    
    x_edge  = np.array([foot_length,foot_length,-foot_length,-foot_length,foot_length])/2
    y_edge  = np.array([foot_width ,-foot_width,-foot_width ,foot_width  ,foot_width])/2
       
    # Empreinte pied gauche
    G_foot_list = [] 
    for i in range(np.shape(G_unique)[1]):
        x_center = G_unique[0,i]
        y_center = G_unique[1,i]
        
        x_tmp = x_edge + x_center
        y_tmp = y_edge + y_center
        
        tmp = np.concatenate((x_tmp, y_tmp))
        tmp = np.reshape(tmp,(2,-1))

        G_foot_list = G_foot_list + [tmp]

        # Empreinte pied gauche
    
    D_foot_list = [] 
    for i in range(np.shape(D_unique)[1]):
        x_center = D_unique[0,i]
        y_center = D_unique[1,i]
        
        x_tmp = x_edge + x_center
        y_tmp = y_edge + y_center
        
        tmp = np.concatenate((x_tmp, y_tmp))
        tmp = np.reshape(tmp,(2,-1))

        D_foot_list = D_foot_list + [tmp]
        
    del G_unique, D_unique, x_edge, y_edge
    del x_center, y_center, x_tmp, y_tmp, tmp
            
    return G_foot_list, D_foot_list
            



def extrem_pos(foot_length, foot_width, vect_pos, cont_G, cont_D):
    """
    Création des valeurs minimale et maximale des positions X et Y des pieds du robot.
    Cette fonction sert à quantifier les contraintes de position pour la résolution
    du problème de minimisation.
    CETTE FONCTION NE FOURNIT PAS LES ENVELOPPES CONVEXES DES PAS A CHAQUE INSTANTS.
    
    Parameters
        foot_length : scalar 
                    Longueur du pied (selon l'axe sagittal X) [m]
        foot_width : scalar 
                    Largeur du pied (selon l'axe transverse Y) [m]
        (vect_pos, cont_G, cont_D)   : cf. commentaire en début de code
        
    Return
        Env_XY : [4xN] numpy array
                Tableau contenant les valeurs extrèmes dans lesquelles doit se trouver
                le centre de pression à chaque instant de la trajectoire contenant 
                N points.
                Env_XY[0,i] : valeurs minimales de X à l'instant i.
                Env_XY[1,i] : valeurs maximales de X à l'instant i.
                Env_XY[2,i] : valeurs minimales de Y à l'instant i.
                Env_XY[3,i] : valeurs maximales de Y à l'instant i.
    """
    
    # 0 - Initialisation
    Env_XY = np.zeros((4,len(cont_G)))
           
    for i in range(len(cont_G)):
        if cont_G[i]=='G_contact':
            if cont_D[i]=='D_contact':      # Cas d'un instant de double contact
                Env_XY[0,i]= min(vect_pos[0,i],vect_pos[2,i]) - foot_length/2   # x_min pied gauche/droit
                Env_XY[1,i]= max(vect_pos[0,i],vect_pos[2,i]) + foot_length/2   # x_max pied gauche/droit
                Env_XY[2,i]= min(vect_pos[1,i],vect_pos[3,i]) - foot_width/2    # y_min pied gauche/droit
                Env_XY[3,i]= max(vect_pos[1,i],vect_pos[3,i]) + foot_width/2    # y_max pied gauche/droit
            else:                           # Cas d'un contact avec le seul pied gauche
                Env_XY[0,i]= vect_pos[0,i] - foot_length/2      # x_min pied gauche
                Env_XY[1,i]= vect_pos[0,i] + foot_length/2      # x_max pied gauche
                Env_XY[2,i]= vect_pos[1,i] - foot_width/2       # y_min pied gauche
                Env_XY[3,i]= vect_pos[1,i] + foot_width/2       # y_max pied gauche
        else:                               # Cas d'un contact avec le seul pied droit
            Env_XY[0,i]= vect_pos[2,i] - foot_length/2          # x_min pied droit
            Env_XY[1,i]= vect_pos[2,i] + foot_length/2          # x_max pied droit
            Env_XY[2,i]= vect_pos[3,i] - foot_width/2           # y_min pied droit
            Env_XY[3,i]= vect_pos[3,i] + foot_width/2           # y_max pied droit
    
    
    del foot_length, foot_width, vect_pos, cont_G, cont_D
    return Env_XY
                
            
                













