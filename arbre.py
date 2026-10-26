import fonctions
import numpy as np
import matplotlib.pyplot as plt

#================== CLASS NOEUD =============

class Noeud:
    def __init__(self, box : list, list_pts : list, niveau : int, indice_local : int, fils1 : "Noeud" = None, fils2 : "Noeud" = None):
        self.box = box
        self.list_pts = list_pts
        self.niveau = niveau
        self.indice_local = indice_local
        self.fils1 = fils1
        self.fils2 = fils2
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

def construction_arbre(Npoints, Nleaf, points):

    def box_initial(points):     
        # initialisation avec le premier point de la liste
        x_min = x_max = points[0][0]
        y_min = y_max = points[0][1]
        
        for point in points:
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

            if Lx >= Ly: #on decoupe selon y
                #fils 1: xmin xmax/2
                #fils 2 : xmax/2 xmax
                #rien ne change selon y
                for pts in noeud.list_pts:
                    xmid = (xmax+xmin)/2
                    if pts[0] <= xmid:
                        list_pts1.append(pts)
                    else:
                        list_pts2.append(pts)

                # creation des points de la boîte
                box_1 = [(xmin, ymin), (xmid, ymin), (xmid, ymax), (xmin, ymax)]
                box_2 = [(xmid, ymin), (xmax, ymin), (xmax, ymax), (xmid, ymax)]

            else:
                for pts in noeud.list_pts:
                    ymid = (ymax+ymin)/2
                    if pts[1] <= ymid:
                        list_pts1.append(pts)
                    else:
                        list_pts2.append(pts)

                box_1 = [(xmin, ymin), (xmax, ymin), (xmax, ymid), (xmin, ymid)]
                box_2 = [(xmin, ymid), (xmax, ymid), (xmax, ymax), (xmin, ymax)]

            fils1 = Noeud(box_1, list_pts1, niveau=noeud.niveau+1, indice_local=0)
            fils2 = Noeud(box_2, list_pts2, niveau=noeud.niveau+1, indice_local=1)

            noeud.fils1 = fils1
            noeud.fils2 = fils2

            iteration(fils1)
            iteration(fils2)

    cluster = []
    racine = Noeud(box_initial(points), points, niveau=0, indice_local=0)
    iteration(racine)

    return cluster

#================== AFFICHAGE TERMINAL =============

def afficher_arbre_terminal(noeud, prefixe=""):
    """
    Affiche la structure hiérarchique de l'arbre dans la console.
    """
    if noeud is None:
        return

    # Détermine l'étiquette selon qu'il s'agisse de la racine ou d'un fils
    if noeud.niveau == 0:
        etiquette = "[Racine]"
    else:
        etiquette = f"[Fils {noeud.indice_local}]"

    # Récupération des dimensions réelles de la boîte du nœud
    xmin = min(p[0] for p in noeud.box)
    xmax = max(p[0] for p in noeud.box)
    ymin = min(p[1] for p in noeud.box)
    ymax = max(p[1] for p in noeud.box)

    # Indicateur si le nœud est une feuille
    statut = "🍃 Feuille" if noeud.is_leaf else "🔀 Nœud interne"

    # Affichage des informations du nœud
    print(
        f"{prefixe}{etiquette} Niv {noeud.niveau} | "
        f"X: [{xmin:.2f}, {xmax:.2f}] | Y: [{ymin:.2f}, {ymax:.2f}] | "
        f"{noeud.n_points} pts | {statut}"
    )

    # Préparation du décalage pour la profondeur suivante
    nouvel_indentation = prefixe + "    └── "

    # Appels récursifs sur les enfants
    if noeud.fils1:
        afficher_arbre_terminal(noeud.fils1, nouvel_indentation)
    if noeud.fils2:
        afficher_arbre_terminal(noeud.fils2, nouvel_indentation)


