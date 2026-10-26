import fonctions
import arbre
import numpy as np
import matplotlib.pyplot as plt

# Parametres
Npoints = 40
a = 1
points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")

Nleaf = 10
    
# Construction
liste_clusters = arbre.construction_arbre(Npoints, Nleaf, points)
noeud_racine = liste_clusters[0]

# Affichage textuel dans le terminal
print("--- STRUCTURE DE L'ARBRE ---")
arbre.afficher_arbre_terminal(noeud_racine)

ind_points =[]
points = []
Test_points =[]


