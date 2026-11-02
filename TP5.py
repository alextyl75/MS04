import fonctions
import arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt

testq1 = False
testq2 = True

# Parametres
Npoints = 200
a = 1
Nleaf = 5

ETA = 3

# Generation du maillage
points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points)

if testq1:
    delta = 0.5
    box1 = [(1 + delta,1+ delta),(1+ delta,2+ delta),(2+ delta,1+ delta),(2+ delta,2+ delta)]
    box2 = [(0,0),(0,1),(1,0),(1,1)]
    noeud1 = arbre_fonctions.Noeud(box1, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    noeud2 = arbre_fonctions.Noeud(box2, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    print(arbre_fonctions.distance_box(noeud1,noeud2))
    print(arbre_fonctions.is_admissible(noeud1,noeud2,eta = 2.))

if testq2:
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)
    arbre_fonctions.afficher_arbre_bloc(arbre_bloc[0])