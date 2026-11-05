import fonctions
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
