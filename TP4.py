import fonctions
import arbre
import numpy as np
import matplotlib.pyplot as plt

# Parametres
Npoints = 100
a = 1
Nleaf = 5

# Generation du maillage
points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    
# Construction de l'arbre de clusters
liste_clusters = arbre.construction_arbre(Npoints, Nleaf, points)
noeud_racine = liste_clusters[0]

# Affichage textuel dans le terminal
print("--- STRUCTURE DE L'ARBRE ---")
arbre.afficher_arbre_terminal(noeud_racine)

# Affichage des clusters
arbre.affichage_clusters(points=points,
    segments=segments,
    liste_clusters=liste_clusters,
    Npoints=Npoints,
    Nleaf=Nleaf,
    a=a,
    afficher_ids=True
)

ind_points =[]
points = []
Test_points =[]





