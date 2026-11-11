
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from typing import List, Optional


#================== CLASS NOEUD =============

class Noeud:
    def __init__(self, box : list, list_pts : list, niveau : int, indice_local : int, fils1 : "Noeud" = None, fils2 : "Noeud" = None, diam : float = 0.):
        self.box = box
        self.list_pts = list_pts
        self.niveau = niveau
        self.indice_local = indice_local
        self.fils1 = fils1
        self.fils2 = fils2
        self.diam = diam
        return 

    @property
    def is_leaf(self):
        """Retourne True si le nœud est une feuille."""
        return self.fils1 is None and self.fils2 is None

    @property
    def n_points(self):
        """Nombre de degrés de liberté dans le cluster."""
        return len(self.list_pts)

#================== ARBRE =============

def construction_arbre(Npoints, Nleaf, points,strat):
    """Renvoie l'arbre sous forme d'une liste de noeud
    """
    def calculer_box_et_diam(pts_indexed):     
        # Calcule la boîte resserrée (Q6) et son diamètre autour d'une liste de (idx, point)
        x_coords = [pt[1][0] for pt in pts_indexed]
        y_coords = [pt[1][1] for pt in pts_indexed]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        box = [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
        diam = float(np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2))
        return box, diam
    
    def box_initial(pts_indexed):     
        # initialisation avec le premier point de la liste
        x_min = x_max = pts_indexed[0][1][0]
        y_min = y_max = pts_indexed[0][1][1]
        
        for idx, point in pts_indexed:
            if point[0]>x_max : x_max = point[0]
            if point[0]<x_min : x_min = point[0]

            if point[1]>y_max : y_max = point[1]
            if point[1]<y_min : y_min = point[1]

        return [[x_min,y_min],[x_min,y_max],[x_max,y_min],[x_max,y_max]]

    def iteration(noeud):
        cluster.append(noeud)

        if len(noeud.list_pts) >= Nleaf:
            xmin = min([pts[0] for pts in noeud.box])
            ymin = min([pts[1] for pts in noeud.box])

            xmax = max([pts[0] for pts in noeud.box])
            ymax = max([pts[1] for pts in noeud.box])

            # Choix de la direction de découpe
            Lx = xmax - xmin
            Ly = ymax - ymin

            list_pts1, list_pts2 = [], []

            if strat=="geo":
                ###########SI DECOUPE GEOMETRIQUE (avant question 6)
                
                if Lx >= Ly: #on decoupe selon y
                    #fils 1: xmin xmax/2
                    #fils 2 : xmax/2 xmax
                    #rien ne change selon y
                    for pts in noeud.list_pts:
                        xmid = (xmax+xmin)/2
                        if pts[1][0] <= xmid:
                            list_pts1.append(pts)
                        else:
                            list_pts2.append(pts)

                    # # creation des points de la boîte
                    # box_1 = [(xmin, ymin), (xmid, ymin), (xmid, ymax), (xmin, ymax)]
                    # box_2 = [(xmid, ymin), (xmax, ymin), (xmax, ymax), (xmid, ymax)]
                    # diam_1 = float(np.sqrt((xmid - xmin)**2 + (ymax - ymin)**2))
                    # diam_2 = float(np.sqrt((xmax - xmid)**2 + (ymax - ymin)**2))

                    box_1, diam_1 = calculer_box_et_diam(list_pts1)
                    box_2, diam_2 = calculer_box_et_diam(list_pts2)


                else:
                    for pts in noeud.list_pts:
                        ymid = (ymax+ymin)/2
                        if pts[1][1] <= ymid:
                            list_pts1.append(pts)
                        else:
                            list_pts2.append(pts)

                    # box_1 = [(xmin, ymin), (xmax, ymin), (xmax, ymid), (xmin, ymid)]
                    # box_2 = [(xmin, ymid), (xmax, ymid), (xmax, ymax), (xmin, ymax)]
                    # diam_1 = float(np.sqrt((xmax - xmin)**2 + (ymid - ymin)**2))
                    # diam_2 = float(np.sqrt((xmax - xmax)**2 + (ymax - ymid)**2))

                    box_1, diam_1 = calculer_box_et_diam(list_pts1)
                    box_2, diam_2 = calculer_box_et_diam(list_pts2)
            if strat=="mediane":
            ############SI DECOUPE MEDIANE (question 6 et suivant)
                if Lx >= Ly: #on decoupe selon y
                    alpha = 0
                else:
                    alpha = 1
                # 1. On trie directement les éléments (idx, coords) selon la coordonnée alpha (pt[1][alpha])
                pts_tries = sorted(noeud.list_pts, key=lambda pt: pt[1][alpha])

                # 2. On coupe la liste triée en deux moitiés égales (médiane)
                mid = len(pts_tries) // 2
                list_pts1 = pts_tries[:mid]
                list_pts2 = pts_tries[mid:]

                # 3. Calcul des nouvelles boîtes englobantes resserrées et de leurs diamètres (Q6)
                box_1, diam_1 = calculer_box_et_diam(list_pts1)
                box_2, diam_2 = calculer_box_et_diam(list_pts2)
        ##################### -----####################------------------------------------------------------

            fils1 = Noeud(box_1, list_pts1, niveau=noeud.niveau+1, indice_local=0,diam = diam_1)
            fils2 = Noeud(box_2, list_pts2, niveau=noeud.niveau+1, indice_local=1, diam =diam_2)

            noeud.fils1 = fils1
            noeud.fils2 = fils2

            iteration(fils1)
            iteration(fils2)

    cluster = []
    # Conversion initiale des points au format (idx, coordonnees)
    points_indexed = [(i, pt) for i, pt in enumerate(points)]
    box, diam = calculer_box_et_diam(points_indexed)
    racine = Noeud(box, points_indexed, niveau=0, indice_local=0,diam=diam)
    iteration(racine)

    return cluster

# Question 5
def hauteur(arbre):
    def exploration(noeud):
        if noeud.is_leaf:
            return noeud.niveau

        prof1 = exploration(noeud.fils1)
        prof2 = exploration(noeud.fils2)

        return max(prof1, prof2)

    return exploration(arbre[0])   

def nombre_feuilles(arbre):
    def exploration(noeud):
        if noeud.is_leaf:
            return 1

        leaf1 = exploration(noeud.fils1)
        leaf2 = exploration(noeud.fils2)

        return leaf1 + leaf2

    return exploration(arbre[0])   

# TP5 question 1

def distance_box(Noeud1,Noeud2):
    x1_min = min(p[0] for p in Noeud1.box)
    x1_max = max(p[0] for p in Noeud1.box)
    y1_min = min(p[1] for p in Noeud1.box)
    y1_max = max(p[1] for p in Noeud1.box)

    x2_min = min(p[0] for p in Noeud2.box)
    x2_max = max(p[0] for p in Noeud2.box)
    y2_min = min(p[1] for p in Noeud2.box)
    y2_max = max(p[1] for p in Noeud2.box)
    
    delta_x = max(0., x1_min - x2_max, x2_min - x1_max)
    delta_y = max(0.,y1_min - y2_max, y2_min - y1_max)

    return np.sqrt(delta_x**2 + delta_y**2)

def affichage_diametre_moyen(arbre,titre):
    diam = 0.
    nb = 0
    for noeud in arbre:
        if noeud.is_leaf:
            diam += noeud.diam
            nb += 1
    print(titre,diam/nb)

def is_admissible(noeudX, noeudY, eta):
    return min(noeudX.diam,noeudY.diam)<= eta * distance_box(noeudX,noeudY)

# Question 2 TP5

class Noeud_bloc:
    def __init__(self, tau : Noeud, sigma : Noeud, niveau : int, indice_local : int, fils_list : Optional[List['Noeud_bloc']] = None, type_feuille : str = None):
        self.tau = tau
        self.sigma = sigma
        self.niveau = niveau
        self.indice_local = indice_local
        self.fils_list = fils_list
        self.type_feuille = type_feuille
        return
        
    @property
    def is_leaf(self):
        """Retourne True si le nœud est une feuille."""
        return self.fils_list is None or len(self.fils_list) == 0

    def diviser(self):
            """Crée les sous-blocs d'interaction enfants valides."""
            enfants_tau = [f for f in (self.tau.fils1, self.tau.fils2) if f is not None]
            enfants_sigma = [f for f in (self.sigma.fils1, self.sigma.fils2) if f is not None]
    
            self.fils_list = [
                Noeud_bloc(
                    tau=child_tau,
                    sigma=child_sigma,
                    niveau=self.niveau + 1,
                    indice_local=i
                )
                for i, (child_tau, child_sigma) in enumerate(
                    (t, s) for t in enfants_tau for s in enfants_sigma
                )
            ]

def arbre_bloc_construction(arbre_cluster, eta):
    arbre_bloc = []

    def iteration(Noeud_bloc):
        arbre_bloc.append(Noeud_bloc)
        
        if is_admissible(Noeud_bloc.tau, Noeud_bloc.sigma, eta): 
            Noeud_bloc.type_feuille="feuille_admissible"
        elif Noeud_bloc.tau.is_leaf or Noeud_bloc.sigma.is_leaf:
            Noeud_bloc.type_feuille="feuille_dense"
        else:      
            Noeud_bloc.diviser()
            for fils in Noeud_bloc.fils_list:
                iteration(fils)

    print(arbre_cluster[0].diam)
    racine_cluster = arbre_cluster[0] if isinstance(arbre_cluster, (list, tuple)) else arbre_cluster
    iteration(Noeud_bloc(racine_cluster, racine_cluster, 0, 0))
    return arbre_bloc

def arbre_bloc_partition(racine_bloc: Noeud_bloc, Npoints: int) -> bool:
    total_couples = 0
    feuilles_parcourues = []

    def parcourt(noeud_bloc):
        nonlocal total_couples
        
        if noeud_bloc.is_leaf:
            if noeud_bloc in feuilles_parcourues:
                raise ValueError(f"Erreur : le nœud {noeud_bloc} a été parcouru deux fois !")
            
            feuilles_parcourues.append(noeud_bloc)
            
            n_tau = len(noeud_bloc.tau.list_pts)
            n_sigma = len(noeud_bloc.sigma.list_pts)
            total_couples += n_tau * n_sigma

        else:
            if noeud_bloc.fils_list:
                for fils in noeud_bloc.fils_list:
                    parcourt(fils)
    parcourt(racine_bloc)

    attendu = Npoints * Npoints
    if total_couples != attendu:
        print(f"Erreur de partition : somme des couples = {total_couples} | attendu = {attendu}")
        return False
    else:
        print(f"Partition valide : {len(feuilles_parcourues)} feuilles couvrent exactement {total_couples} couples ({Npoints}x{Npoints}).")
        return True 


def H_matrice_gain_memoire(A_full, arbre_bloc, aca_function, tol):
    N_h = 0

    def parcourt(Noeud_bloc):
        nonlocal N_h

        if Noeud_bloc.is_leaf:
            tau = Noeud_bloc.tau
            sigma = Noeud_bloc.sigma
            
            if Noeud_bloc.type_feuille == "feuille_admissible":
                idx_tau = [p[0] for p in tau.list_pts]
                idx_sigma = [p[0] for p in sigma.list_pts] 
                A_ij = A_full[np.ix_(idx_tau, idx_sigma)]

                _, _, r = aca_function(A_ij, tol)
                N_h += r * (len(sigma.list_pts) + len(tau.list_pts))

            else: 
                N_h += len(sigma.list_pts) * len(tau.list_pts)
        else:
            for fils in Noeud_bloc.fils_list:
                parcourt(fils)

    parcourt(arbre_bloc[0])

    return N_h

def H_matrix(A_full, arbre_bloc, aca_function, tol):
    """Construit la H-matrice sous forme de dictionnaire.
    Clé   : Tuple (tuple(idx_tau), tuple(idx_sigma))
    Valeur: - (U, V) pour un bloc admissible (matrices NumPy de rang faible)
            - A_ij   pour un bloc dense
    """
    H = {}

    def parcourt(Noeud_bloc):
        nonlocal H

        if Noeud_bloc.is_leaf:
            tau = Noeud_bloc.tau
            sigma = Noeud_bloc.sigma

            # Extraction des indices
            idx_tau = [p[0] for p in tau.list_pts]
            idx_sigma = [p[0] for p in sigma.list_pts]

            # Clé unique pour le dictionnaire
            cle = (tuple(idx_tau), tuple(idx_sigma))
            A_ij = A_full[np.ix_(idx_tau, idx_sigma)]

            if Noeud_bloc.type_feuille == "feuille_admissible":
                u_list, v_list, r = aca_function(A_ij, tol)

                if r > 0:
                    U = np.column_stack(u_list)
                    V = np.column_stack(v_list)
                else:
                    U = np.zeros((len(idx_tau), 0), dtype=complex)
                    V = np.zeros((len(idx_sigma), 0), dtype=complex)

                H[cle] = (U, V)

            elif Noeud_bloc.type_feuille == "feuille_dense":
                H[cle] = A_ij

        else:
            for fils in Noeud_bloc.fils_list:
                parcourt(fils)

    racine = arbre_bloc[0] if isinstance(arbre_bloc, list) else arbre_bloc
    parcourt(racine)

    return H

def H_matrix_assemblage(H, Npoints):
    """
    Reconstitue la matrice dense H (Npoints x Npoints) à partir du dictionnaire H.
    """
    H_assemble = np.zeros((Npoints, Npoints), dtype=complex)

    for (idx_tau, idx_sigma), contenu in H.items():
        indices = np.ix_(idx_tau, idx_sigma)
        
        if isinstance(contenu, tuple):
            # Bloc admissible : contenu = (U, V)
            U, V = contenu            
            H_assemble[indices] = U @ V.T
        else:
            # Bloc dense : contenu = A_ij
            H_assemble[indices] = contenu

    return H_assemble

def Hmatvec(H, b):
    # Forcer le type complexe (ou utiliser le type de b s'il est déjà complexe)
    dtype = np.result_type(b.dtype, np.complex128)
    result = np.zeros(b.size, dtype=dtype)

    for (idx_tau, idx_sigma), contenu in H.items():
        tau = list(idx_tau)
        sigma = list(idx_sigma)

        if isinstance(contenu, tuple):
            U, V = contenu
            v_b = V.T @ b[sigma]
            result[tau] += U @ v_b
        else:
            result[tau] += contenu @ b[sigma]

    return result



def H_matrix_light(
    arbre_bloc,
    aca_function,
    tol,
    points,
    segments,
    k,
    x_quad,
    w,
    x_quad_tilde,
    w_tilde,
    fonctions=None,
):
    """Construit la H-matrice sous forme de dictionnaire sans jamais assembler la matrice complète.
    Clé   : Tuple (tuple(idx_tau), tuple(idx_sigma))
    Valeur: - (U, V) pour un bloc admissible (matrices NumPy de tailles (N, r) et (M, r) avec A_bloc ≈ U @ V.T)
            - A_ij   pour un bloc dense (matrice NumPy de taille (N, M))
    """
    H = {}

    def parcourt(Noeud_bloc):
        if Noeud_bloc.is_leaf:
            tau = Noeud_bloc.tau
            sigma = Noeud_bloc.sigma

            # Extraction des indices globaux des segments du bloc
            idx_tau = [p[0] for p in tau.list_pts]
            idx_sigma = [p[0] for p in sigma.list_pts]

            cle = (tuple(idx_tau), tuple(idx_sigma))
            indices = np.ix_(idx_tau, idx_sigma)

            if Noeud_bloc.type_feuille == "feuille_admissible":
                u_list, v_list, r = aca_function(
                    points, segments, k, indices, tol, x_quad, w, x_quad_tilde, w_tilde, fonctions
                )

                if r > 0:
                    U = np.column_stack(u_list)
                    V = np.column_stack(v_list)
                else:
                    U = np.zeros((len(idx_tau), 0), dtype=complex)
                    V = np.zeros((len(idx_sigma), 0), dtype=complex)

                H[cle] = (U, V)

            elif Noeud_bloc.type_feuille == "feuille_dense":
                H[cle] = A_assemble_bloc(
                    points, segments, indices, x_quad, w, x_quad_tilde, w_tilde, k, fonctions
                )
        else:
            for fils in Noeud_bloc.fils_list:
                parcourt(fils)

    racine = arbre_bloc[0] if isinstance(arbre_bloc, list) else arbre_bloc
    parcourt(racine)

    return H

######################
#Pour matrice hiérarchiques



def A_assemble_bloc(points, segments, indices, x_quad, w, x_quad_tilde, w_tilde, k, fonctions=None):
    """Assemble un bloc dense de la matrice BEM correspondant aux indices (idx_tau, idx_sigma)."""
    idx_tau = np.asarray(indices[0]).ravel()
    idx_sigma = np.asarray(indices[1]).ravel()
    N, M = len(idx_tau), len(idx_sigma)

    nq = len(x_quad)
    nq_tilde = len(x_quad_tilde)
    G_func = fonctions

    A = np.zeros((N, M), dtype=complex)
    gamma_euler = 0.5772156649

    # Pré-calcul des poids combinés
    W_reg_matrix = np.outer(w, w_tilde)  # shape: (nq, nq_tilde)
    somme_w_reg = np.sum(W_reg_matrix)
    const_reg = 1j / 4 - (1 / (2 * np.pi) * (np.log(k / 2) + gamma_euler))

    for i in range(N):
        ig = idx_tau[i]  # Indice global du segment ligne
        ai, bi = points[segments[ig][0]], points[segments[ig][1]]
        vec_i = bi - ai
        length_i = np.linalg.norm(vec_i)
        demi_norm_i = 0.5 * length_i

        # Points de quadrature sur le segment ig : shape (nq, 2)
        x_pts = 0.5 * (ai + bi) + 0.5 * np.outer(x_quad, vec_i)

        for j in range(M):
            jg = idx_sigma[j]  # Indice global du segment colonne

            if ig == jg:
                # --- CAS DIAGONAL GLOBAL (Singularité : même segment) ---
                reg = somme_w_reg * const_reg * (demi_norm_i ** 2)

                Tau_e = vec_i / length_i
                sing = 0.0
                for k_index in range(nq):
                    d_b_x = bi - x_pts[k_index]
                    d_a_x = ai - x_pts[k_index]

                    term = (
                        np.dot(d_b_x, Tau_e) * np.log(np.linalg.norm(d_b_x))
                        - np.dot(d_a_x, Tau_e) * np.log(np.linalg.norm(d_a_x))
                        - length_i
                    )
                    sing += w[k_index] * term

                sing *= -(1 / (2 * np.pi)) * demi_norm_i
                A[i, j] = sing + reg

            else:
                # --- CAS HORS-DIAGONALE ---
                aj, bj = points[segments[jg][0]], points[segments[jg][1]]
                demi_norm_j = 0.5 * np.linalg.norm(bj - aj)

                # Points de quadrature sur le segment jg : shape (nq_tilde, 2)
                y_pts = 0.5 * (aj + bj) + 0.5 * np.outer(x_quad_tilde, bj - aj)

                Aij = 0.0j
                for k_index in range(nq):
                    for k_tilde_index in range(nq_tilde):
                        val_G = G_func(x_pts[k_index], y_pts[k_tilde_index], k)
                        Aij += W_reg_matrix[k_index, k_tilde_index] * val_G

                A[i, j] = Aij * demi_norm_i * demi_norm_j

    return A


def calcul_ligne(points, segments, indices, i, x_quad, w, x_quad_tilde, w_tilde, k, fonctions=None):
    """Calcule la i-ème ligne locale du bloc défini par 'indices'."""
    idx_tau = np.asarray(indices[0]).ravel()
    idx_sigma = np.asarray(indices[1]).ravel()
    M = len(idx_sigma)

    G_func = fonctions
    ligne = np.zeros(M, dtype=complex)

    # Indice global de la ligne i
    ig = idx_tau[i]
    ai = points[segments[ig][0]]
    bi = points[segments[ig][1]]
    vec_i = bi - ai
    norm_i = np.linalg.norm(vec_i)
    demi_norm_i = 0.5 * norm_i

    pts_i = 0.5 * (ai + bi) + 0.5 * np.outer(x_quad, vec_i)

    gamma_euler = 0.5772156649
    constante_reg = 4 * (1j / 4 - (1 / (2 * np.pi) * (np.log(k / 2) + gamma_euler))) * (demi_norm_i ** 2)

    for j in range(M):
        jg = idx_sigma[j]  # Indice global de la colonne j

        if ig == jg:
            # --- TERME SINGULIER (Diagonale globale) ---
            sing = 0.0
            Tau_e = vec_i / norm_i

            for k_idx in range(len(x_quad)):
                d_b_x = bi - pts_i[k_idx]
                d_a_x = ai - pts_i[k_idx]

                terme = (
                    np.dot(d_b_x, Tau_e) * np.log(np.linalg.norm(d_b_x))
                    - np.dot(d_a_x, Tau_e) * np.log(np.linalg.norm(d_a_x))
                    - norm_i
                )
                sing += w[k_idx] * terme

            sing *= -(1 / (2 * np.pi)) * demi_norm_i
            ligne[j] = sing + constante_reg

        else:
            # --- TERME RÉGULIER (Hors-diagonale) ---
            aj = points[segments[jg][0]]
            bj = points[segments[jg][1]]
            demi_norm_j = 0.5 * np.linalg.norm(bj - aj)

            pts_j = 0.5 * (aj + bj) + 0.5 * np.outer(x_quad_tilde, bj - aj)

            Aij = 0.0j
            for k_idx in range(len(x_quad)):
                for k_tilde_idx in range(len(x_quad_tilde)):
                    Aij += w[k_idx] * w_tilde[k_tilde_idx] * G_func(pts_i[k_idx], pts_j[k_tilde_idx], k)

            ligne[j] = Aij * demi_norm_i * demi_norm_j

    return ligne


def calcul_colonne(points, segments, indices, j, x_quad, w, x_quad_tilde, w_tilde, k, fonctions=None):
    """Calcule la j-ème colonne locale du bloc défini par 'indices'."""
    idx_tau = np.asarray(indices[0]).ravel()
    idx_sigma = np.asarray(indices[1]).ravel()
    N = len(idx_tau)

    G_func = fonctions
    colonne = np.zeros(N, dtype=complex)

    # Indice global de la colonne j
    jg = idx_sigma[j]
    aj = points[segments[jg][0]]
    bj = points[segments[jg][1]]
    vec_j = bj - aj
    norm_j = np.linalg.norm(vec_j)
    demi_norm_j = 0.5 * norm_j

    pts_j = 0.5 * (aj + bj) + 0.5 * np.outer(x_quad_tilde, vec_j)

    gamma_euler = 0.5772156649

    for i in range(N):
        ig = idx_tau[i]  # Indice global de la ligne i

        if ig == jg:
            # --- TERME SINGULIER (Diagonale globale) ---
            Tau_e = vec_j / norm_j
            sing = 0.0
            pts_i = 0.5 * (aj + bj) + 0.5 * np.outer(x_quad, vec_j)

            for k_idx in range(len(x_quad)):
                d_b_x = bj - pts_i[k_idx]
                d_a_x = aj - pts_i[k_idx]

                terme = (
                    np.dot(d_b_x, Tau_e) * np.log(np.linalg.norm(d_b_x))
                    - np.dot(d_a_x, Tau_e) * np.log(np.linalg.norm(d_a_x))
                    - norm_j
                )
                sing += w[k_idx] * terme

            sing *= -(1 / (2 * np.pi)) * demi_norm_j
            constante_reg = 4 * (1j / 4 - (1 / (2 * np.pi) * (np.log(k / 2) + gamma_euler))) * (demi_norm_j ** 2)
            colonne[i] = sing + constante_reg

        else:
            # --- TERME RÉGULIER (Hors-diagonale) ---
            ai = points[segments[ig][0]]
            bi = points[segments[ig][1]]
            demi_norm_i = 0.5 * np.linalg.norm(bi - ai)

            pts_i = 0.5 * (ai + bi) + 0.5 * np.outer(x_quad, bi - ai)

            Aij = 0.0j
            for k_idx in range(len(x_quad)):
                for k_tilde_idx in range(len(x_quad_tilde)):
                    Aij += w[k_idx] * w_tilde[k_tilde_idx] * G_func(pts_i[k_idx], pts_j[k_tilde_idx], k)

            colonne[i] = Aij * demi_norm_i * demi_norm_j

    return colonne


def ACA_partiel_free_bloc(points, segments, k, indices, epsilon, x_quad, w, x_quad_tilde, w_tilde, fonctions=None):
    """ACA partiel sans assemblage complet (matrix-free).
    Renvoie (u_k_used, v_k_used, r) tels que A_bloc ≈ U @ V.T
    """
    idx_tau = np.asarray(indices[0]).ravel()
    idx_sigma = np.asarray(indices[1]).ravel()
    N, M = len(idx_tau), len(idx_sigma)
    max_rank = min(N, M)

    A_approx_norm = 0.0
    next_indice = 0
    index_used = []
    cols_used = []
    v_k_used = []
    u_k_used = []
    norm_prod_uv = 0.0

    def ACA_partiel_step(index_ligne):
        # 1. Calcul de la ligne résiduelle R[index_ligne, :]
        res_row = calcul_ligne(
            points, segments, indices, index_ligne, x_quad, w, x_quad_tilde, w_tilde, k, fonctions
        )
        for u, v in zip(u_k_used, v_k_used):
            res_row -= u[index_ligne] * v

        # 2. Recherche du pivot maximal dans la ligne résiduelle
        abs_res_row = np.abs(res_row)
        abs_res_row[cols_used] = -1.0
        j = int(np.argmax(abs_res_row))
        pivot = res_row[j]

        index_used.append(index_ligne)

        if np.abs(pivot) < 1e-12:
            u_k = np.zeros(N, dtype=complex)
            v_k = np.zeros(M, dtype=complex)
            # Choisir une prochaine ligne non encore visitée si possible
            remaining_rows = [ idx for idx in range(N) if idx not in index_used ]
            next_indice_val = remaining_rows[0] if remaining_rows else 0
        else:
            # 3. Calcul de la colonne résiduelle R[:, j]
            res_col = calcul_colonne(
                points, segments, indices, j, x_quad, w, x_quad_tilde, w_tilde, k, fonctions
            )
            for u, v in zip(u_k_used, v_k_used):
                res_col -= u * v[j]

            u_k = res_col / pivot
            v_k = res_row
            cols_used.append(j)

            # 4. Prochaine ligne pivot choisie sur le maximum de |u_k|
            col_abs = np.abs(u_k).copy()
            col_abs[index_used] = -1.0
            next_indice_val = int(np.argmax(col_abs))

        u_k_used.append(u_k)
        v_k_used.append(v_k)
        return next_indice_val

    for step in range(1, max_rank + 1):
        next_indice = ACA_partiel_step(next_indice)

        u_k = u_k_used[-1]
        v_k = v_k_used[-1]
        norm_prod_uv = np.linalg.norm(u_k) * np.linalg.norm(v_k)

        if norm_prod_uv == 0.0:
            u_k_used.pop()
            v_k_used.pop()
            break

        # Mise à jour récursive de la norme de Frobenius ||A_k||_F
        sum_norm = 0.0j
        for l in range(len(u_k_used) - 1):
            sum_norm += np.vdot(u_k_used[l], u_k) * np.vdot(v_k_used[l], v_k)

        valeur_sous_racine = A_approx_norm ** 2 + 2.0 * sum_norm.real + norm_prod_uv ** 2
        A_approx_norm = np.sqrt(max(0.0, valeur_sous_racine))

        # Critère d'arrêt relatif
        if norm_prod_uv < epsilon * A_approx_norm:
            break

    r = len(u_k_used)
    return u_k_used, v_k_used, r

