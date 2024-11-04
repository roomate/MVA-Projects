# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt



# ---------------------------------------------------------------
# 1 - Construction des matrices Pu et Px: solution analytique
#----------------------------------------------------------------
def Construction_matrix(N,Dt,h,g):
    """
    Création des matrices de récurrence liant les itérations de la position du Centre de Pression (z(k)) avec les itérations 
    du vecteur (x^(k))  et les itérations du vecteur de commande (x...(k) : le jerk).
    Cette fonction retranscrit la formule (8) de l'article [R1] 
    
    Paramètres
        N : scalaire
            Nombre de pas de planification pris en compte dans la matrice (horizon de contrôle) [s.u]
        T : scalaire
            Pas de planification, i.e. pas d'échantilonnage de la matrice [s]
        h : scalaire
            Différence de hauteur entre le Centre de Gravité et le Centre de Pression [m]
        g : scalaire
            Constante de gravitation terrestre [m/s²]

    Returns
        Px  : [Nx3] array
                Matrice sous forme de tableau liant le vecteur (Z(k+1) : [Nx1]) avec les itérés du vecteur (x^(k)   : [3x1])
        Pu  : [NxN] array
                Matrice sous forme de tableau liant le vecteur (Z(k+1) : [Nx1]) avec les itérés du vecteur (x...(k) : [Nx1])
    """
    # Initialisation
    P_x = []
    P_u = np.zeros((N,N))

    # Création de P_x
    for i in range(N): 
        P_x += [[1, (i+1)*Dt, (i+1)**2*Dt**2/2 - h/g]]
    P_x = np.array(P_x.copy())

    # Création de P_u
    for i in range(N):
        tmp_val = (1+3*i+3*i**2)*(Dt**3)/6 - Dt*h/g
        if i ==0:
            tmp_arr = np.ones((1,1))*tmp_val
        else:
            tmp_arr = np.insert(tmp_arr,0,tmp_val)
            tmp_arr = np.reshape(tmp_arr,(1,-1))

        P_u[i,0:np.size(tmp_arr)] = tmp_arr
       
    del N,Dt,h,g 

    return P_x, P_u



# ---------------------------------------------------------------
# 1 - Construction des matrices A et B du modèle d'état
#----------------------------------------------------------------
def Compute_A_B(Dt):
    """
    La fonction fournit les matrices A et B du modèle d'état (équation (3)).
        
    Paramètres
        Dt : scalaire
            Valeur du pas de temps considéré [s]
            
    Returns
        A   : [3x3] array
            Matrice de transition
        B   : [3x1] array
            Matrice de commande
    """
    A = np.array([[1, Dt, (Dt**2)/2], [0,1,Dt], [0,0,1]])
    B = np.array([(Dt**3)/6, (Dt**2)/2, Dt])
    B = np.reshape(B,(3,1))
    
    del Dt
    return A,B



# ---------------------------------------------------------------
# 2 - Propagation du modèle d'état 
#----------------------------------------------------------------
def Compute_State(A, B, X_k, DX_k, U_k):
    """
    La fonction calcule un itéré du modèle d'état suivant : 
        X(k+1) = [A].X(k) + [B].U(k) + DX(k)
    Le lien entre itéré et durée considérée pour l'itération est implicitement introduite dans
    les matrices [A] et [B].
    
    Paramètres
        A   : [3x3] array
            Matrice de transition
        B   : [3x1] array
            Matrice de commande
        X_k : [3x1] array
            Vecteur d'état à l'instant k (Pos(k), Vit(k), Acc(k)).T   [SI]
        DX_k: [3x1] array
            Vecteur de perturbtaion d'état à l'instant k (DPos(k), DVit(k), DAcc(k)).T   [SI]
        U_k : scalaire
            Commande à l'instant k (le jerk dans notre cas)      [SI]

    Returns
        X_new  : [3x1] array
            Vecteur d'état à l'instant k+1, issu du modèle présenté plus haut
    
    """
    # Mise à la bonne dimension
    X_k  = np.reshape(X_k,(3,1))
    DX_k = np.reshape(DX_k,(3,1))
    U_k  = np.reshape(U_k,(1,1))
    
    # Calcul de l'état k+1    
    X_new = np.dot(A,X_k) + np.dot(B,U_k) + DX_k
    
    del A, B, X_k, DX_k, U_k
    return X_new


# ---------------------------------------------------------------
# 4 - Calcul de la commande (Jerk) : Solution analytique
#----------------------------------------------------------------
def Compute_command_ana(X_k, Z_ref,Px, Pu, ratio):
    """
    Création de la commande (jerk) analytique.
    Cette fonction retranscrit les formules (11) et (12) de l'article [R1].
    L'horizon de temps sur laquelle la solution est calculée est dénotée N (nombre d'itérations). 
    Il s'agit d'une composante implicite, celle-ci étant contenue dans la dimension des matrices Px, Pu et du vecteur Z_ref.
    
    Paramètres
        X_k  : [3x1] array
                Vecteur d'état à l'instant k (Pos(k), Vit(k), Acc(k)).T   [SI]
        Z_ref: [Nx1] array
                Trajectoire de référence du Centre de pressionnde de l'instant (k) à (k+N-1)  [m]
        Px  : [Nx3] array
                Matrice sous forme de tableau liant le vecteur (Z(k+1) : [Nx1]) avec les itérés du vecteur (x^(k)   : [3x1])
        Pu  : [NxN] array
                Matrice sous forme de tableau liant le vecteur (Z(k+1) : [Nx1]) avec les itérés du vecteur (x...(k) : [Nx1])
        ratio : scalaire
            Relation R/Q entre le poids affecté à la commande (R) et l'écart de suivi du centre de pression (Q) [s.u]

    Returns
        Jerk_out  : [Nx1] array
                Succession de N commande (jerk) de l'instant (k+1) à (k+N) (cf. équation (11))
        U_k       : scalaire
                Valeur de la commande à considérer, correspondant à la première composante du vecteur "Jerk_out" (cf. équation (12))
    """
    
    # Mise à la bonne dimension
    X_k     = np.reshape(X_k,(-1,1))
    Z_ref   = np.reshape(Z_ref,(-1,1))

    # Calcul de la commande
    Mat = np.dot(np.transpose(Pu),Pu) + ratio*np.eye(np.shape(Px)[0])
    Mat = np.linalg.inv(Mat)
    Mat = np.dot(Mat,np.transpose(Pu))
    
    DZ = np.dot(Px,X_k) - Z_ref
    
    Jerk_out = -np.dot(Mat,DZ)
    u_k      = Jerk_out[0,0]
        
    del Px, Pu, ratio, X_k, Z_ref 
    del Mat, DZ
    return Jerk_out, u_k





# ---------------------------------------------------------------
# 5 - Simulation de la loi de contrôle analytique
#----------------------------------------------------------------
def Control_ana(x_0, Z_ref, Vect_tps,hori,h,g,ratio):
    """
    Cette fonction exécute la loi de contrôle analytique pendant une séquence 
    de marche définie par le vecteur "Z_ref". Le vecteur temps de la simulation est
    défini par "Vect_tps". On met en avant les points suivants:
            - le pas de calcul est calculé avec le vecteur "Vect_tps"; ainsi il
            est implicite que ce vecteur est à pas constant
            - La durée de simulation est prolongée d'une durée "hori" 
            pour fournir une loi de commande couvrant l'entièreté de la durée définie
            par "Vect_tps". On prolonge la dernière valeur du vecteur "Z_ref."
            - Le paramètre "hori" est l'horizon de temps sur lequel la loi de commande
            est calculée.
     IMPORTANT : 
     - Entre chaque itération de la loi de commande, on calcule le vecteur
     d'état du système avec une fréquence 5 fois plus rapide. 
     Cela permet de modéliser la fréquence d'acquisition des capteur et doit
     permettre de créer une léger écart entre l'effet anticipé de la commande
     et son effet réel. CE RATIO DE 1/5 EST SAISI EN DUR POUR LE MOMENT : "x_ratio"
     - On n'a pas implémenté de perturbation de l'état : Dx_k= np.zeros((3,1))
         
            
     Paramètres
         x_0  : [3x1] array
                 Vecteur d'état à l'instant initial 
                 (Pos(0), Vit(0), Acc(0)).T   [SI]
         Z_ref: [Nx1] array
                 Trajectoire de référence du Centre de pression 
                 de de l'instant (k) à (k+N-1)  [m]
         Vect_tps  : [Nx1] array   
                 Vecteur temps de la simulation [s]
         hori : scalaire
                 Horizon de temps (i.e. durée) sur laquelle est calculée la 
                 commande u(k) à l'instant k [s]
         h    : scalaire
                 Différence de hauteur entre le Centre de Gravité et le Centre de Pression [m]
         g    : scalaire
             Constante de gravitation terrestre [m/s²]    
         ratio: scalaire
             Relation R/Q entre le poids affecté à la commande (R) et l'écart de suivi du centre de pression (Q) [s.u]

     Returns
         U  : [Nx1] array
            Tableau contenant les commandes analytiques sur toute la durée de la simulation
        X_hat : [Nx3 array]
            Tableau contenant les trois états du système sur toute la durée de la simulation
        Z   : [Nx1] array
            Tableau contenant les positions du centre de pression sur toute la durée de la simulation
    """
    
    
    # 0 -!!  ECRITURE EN DUR DE PARAMETRES DE REGLAGE !!
    x_ratio = 5                 # Rapport de fréquence de calcul des états
    Dx_k    = np.zeros((3,1))   # vecteur des perturbations d'état
    C       = np.array([1, 0, -h/g])
    C       = np.reshape(C,(1,-1))
    
    # 0 - Initialisation des paramètres et tests
    x_0      = np.reshape(x_0,(3,1))
    Z_ref    = np.reshape(Z_ref,(-1,1))
    Vect_tps = np.reshape(Vect_tps,(-1,1))
    ind_end  = np.size(Vect_tps)             # A cause de l'horizon de calcul de U
    Dt       = Vect_tps[1,0]-Vect_tps[0,0]
    
    Flag_size = False
    if np.size(Z_ref)==np.size(Vect_tps):
        Flag_size = True
    
    # 0 - Initialisation des sorties
    U       = np.zeros((1,1))
    X_hat   = x_0
    Z       = np.ones(1,)*h
    del x_0
    
    
    
    # 1 - Prolongation de Z_ref et du vecteur temps (contrainte d'horizon de U)
    Vect_tps_2 = np.linspace(0,(hori-1)*Dt,hori) + Vect_tps[-1] + Dt
    Vect_tps_2 = np.reshape(Vect_tps_2,(-1,1))
    Z_ref_2    = np.ones(np.shape(Vect_tps_2))*Z_ref[-1]
    
    Z_ref      = np.append(Z_ref,Z_ref_2,axis=0)
    Vect_tps   = np.append(Vect_tps,Vect_tps_2,axis=0)
    
    del Vect_tps_2, Z_ref_2
   
    # 2 - Création des paramètres constants
    P_x, P_u = Construction_matrix(hori,Dt,h,g)
    A,B      = Compute_A_B(Dt/x_ratio)
    del h, g
    
    # 3 - BOUCLE DE CONTROLE
    for i in range(ind_end):        # Calcul de la commande
        if Flag_size==False:        # Sécurité pour ne pas commencer un calcul qui n'aboutira pas
            break
        _, u_new = Compute_command_ana(X_hat[:,-1], 
                                       Z_ref[i:i+hori,0],
                                       P_x, P_u, ratio)
        x_new = X_hat[:,-1]
        for j in range(x_ratio):   #Simulation de l'état du système (plus rapide) 
            x_new   = Compute_State(A, B, x_new, Dx_k, u_new)
    
        U     = np.append(U,u_new)
        X_hat = np.append(X_hat,x_new,axis=1)
        z     = np.dot(C,x_new)
        Z     = np.append(Z, z)
    
    del u_new, x_new, z
    del P_x, P_u, A, B
    del ind_end, Dt
    del Z_ref, Vect_tps,hori,ratio

    # XX - Sorties
    return U, X_hat.T, np.reshape(Z,(-1,1))



# ---------------------------------------------------------------
# 6 - Simulation de la loi de contrôle analytique
#----------------------------------------------------------------
def Eigen_max(N_vector,R_Q_vector,Dt,h,g):
    """
    Cette fonction calcule le plus grand module des trois vecteurs propres de 
    la matrice donnée en Equ.14 du docment de référence #1.
    La fonction est faite pour minimiser le temps de calcul en réduisant au mieux
    le nombre d'opérations matricielles'
         
     Paramètres
        R_Q_vector : [kx1] array   
                Vecteur des valeurs du ration R/Q considérées pour le calcul des valeurs propres [s.u]
        N_vector  : [mx1] array   
                 Vecteur des valeurs de N (horizon de la loi de commande en nombre de pas d'itération)
                considérées pour le calcul des valeurs propres [s.u]
         Dt : scalaire
                 Pas de temps utilisé pour le calcul de la loi de commande [s]
         h    : scalaire
                 Différence de hauteur entre le Centre de Gravité et le Centre de Pression [m]
         g    : scalaire
             Constante de gravitation terrestre [m/s²]    
         ratio: scalaire
             Relation R/Q entre le poids affecté à la commande (R) et l'écart de suivi du centre de pression (Q) [s.u]

     Returns
         Lambda_vector  : [kxm] array
            Tableau contenant les plus grands modules des valeurs propres.
            Un module supérieur à 1 indique une loi de commande instable.
            Axe 0 (ligne)  : valeurs du ratio R/Q
            Axe 1 (colone) : valeurs N du nombre d'itérations considérées
    """
  
    # 0 - INITIALISATION
    #-------------------
    Lambda_vector   = np.empty((np.size(R_Q_vector),np.size(N_vector)))   # Pour stocker les valeurs propres
  
    
    # 1 - Création des données constantes
    #------------------------------------
    A,B = Compute_A_B(Dt)
    
    
    # 2 - Réalisation de la fonction
    #------------------------------------
    j=0
    for N in N_vector:                  # Boucle sur le paramètre N*Dt   
        P_x, P_u = Construction_matrix(N,Dt,h,g)
        
        P_uT  = np.transpose(P_u)
        Mat_1 = np.dot(P_uT,P_x)
        Mat_2 = np.dot(P_uT,P_u) 
        
        i=0
        for ratio in R_Q_vector:        # Boucle sur le paramètre R/Q        
            Mat_3 = Mat_2 + ratio*np.eye(N)
            Mat_3 = np.linalg.inv(Mat_3)
            Mat_4 = np.dot(Mat_3,Mat_1)
            
            Mat_f       = A-B*Mat_4[0,0]
            Val_P,_     = np.linalg.eig(Mat_f)
            Val_P_max   = max(np.abs(Val_P))
                    
            Lambda_vector[i,j] = Val_P_max
            i=i+1
            
            del Mat_3, Mat_4, Mat_f, Val_P, Val_P_max
        j = j+1    
        del i, P_x, P_u, P_uT, Mat_1, Mat_2
    del j
    Lambda_vector = np.array(Lambda_vector)
    
    return Lambda_vector   # ligne : R/Q ; col : N







# # ---------------------------------------------------------------
# # XX - ZONE DE SCRIPT POUR TEST DES FONCTIONS
# #----------------------------------------------------------------

# # 00 - INITIALISATION DES PARAMETRES
# #-----------------------------------
# D_simu   = 10      # Durée de la simulation [s]
# Dt       = 0.005   # Intervalle entre échantillon [s]
# n_simu   = int(D_simu/Dt)
# Vect_tps = np.linspace(0,D_simu,n_simu+1)

# hori    = 300  # horizon de calcul de la loi de commande [nbr échantillon]

# h       = 0.8       # hauteur du CdG
# g       = 9.81      # constante de gravité [m/s²]
# ratio   = 10e-6    # ration R/G

# x_0     = np.zeros((3,1))   # Vecteur d'état à l'instant initial

# # 0 - Création du vecteur Z_ref clear
# #------------------------------
# # fonction sinus
# Z_ampl      = 0.3
# Z_freq      = 1  # Hz
# Z_R_sin     = Z_ampl * np.sin(2 * np.pi * Z_freq * Vect_tps)

# # 0 - Création du vecteur Z_ref : fonctions échelons
# n_step      = 4
# Z_R_ech     = [0]*int(0.5*1/Dt) \
#             + ([-Z_ampl]*int(0.5*1/Dt) + [Z_ampl]*int(0.5*1/Dt))*n_step \
#             + [0]*int(0.5*1/Dt) \
#             + [0]
# Vect_tps_ech= list(range(0, np.size(Z_R_ech), 1))
# Vect_tps_ech = np.array(Vect_tps_ech)*Dt
# Vect_tps_ech = np.reshape(Vect_tps_ech,(-1,1))


# # # 1 - Loi de commande et affichage : Z_REF sinusoïdale
# # #------------------------------------------------------------
# # # U_sin, X_hat_sin, Z_sin = Control_ana(x_0, Z_R_sin, Vect_tps,hori,h,g,ratio)
# # # fig, ax = plt.subplots(4, 1, figsize=(10, 5))
# # # ax[0].plot(Vect_tps,Z_R_sin,label='Z_REF')
# # # ax[0].legend(loc = 'best')

# # # ax[1].plot(Z_sin[1:,:],label='Z_est')
# # # ax[1].legend(loc = 'best')

# # # ax[2].plot(U_sin,label='Commande analytique')
# # # ax[2].legend(loc = 'best')

# # # ax[3].plot(X_hat_sin[:,0],label='Position CdG')
# # # ax[3].legend(loc = 'best')

# # # plt.show

# # 1 - Loi de commande et affichage : Z_REF échelon
# #------------------------------------------------------------
# U_ech, X_hat_ech, Z_ech = Control_ana(x_0, Z_R_ech, Vect_tps_ech,hori,h,g,ratio)
# fig, ax = plt.subplots(4, 1, figsize=(10, 5))
# ax[0].plot(Vect_tps_ech,Z_R_ech,label='Z_REF')
# ax[0].legend(loc = 'best')

# ax[1].plot(Z_ech[1:,:],label='Z_est')
# ax[1].legend(loc = 'best')

# ax[2].plot(U_ech,label='Commande analytique')
# ax[2].legend(loc = 'best')

# ax[3].plot(X_hat_ech[:,0],label='Position CdG')
# ax[3].legend(loc = 'best')

# plt.show








