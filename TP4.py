import fonctions
import arbre
import numpy as np
import matplotlib.pyplot as plt

test_affichage_terminal = False
test_affichage_plt = False

test_q3 = True

# Parametres
Npoints = 100
a = 1
Nleaf = 5

# Generation du maillage
points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    
# Construction de l'arbre de clusters

# Question 1 et 2
liste_clusters = arbre.construction_arbre(Npoints, Nleaf, points)
noeud_racine = liste_clusters[0]

# Affichage textuel dans le terminal
if test_affichage_terminal:
    print("--- STRUCTURE DE L'ARBRE ---")
    arbre.afficher_arbre_terminal(noeud_racine)

# Affichage des clusters
if test_affichage_plt:
    arbre.affichage_clusters(points=points,
        segments=segments,
        liste_clusters=liste_clusters,
        Npoints=Npoints,
        Nleaf=Nleaf,
        a=a,
        afficher_ids=True
    )

# Question 3 (test de validité)
if test_q3:
    print("Test de validité : clusters cercle vs maillage cercle ")
    arbre.test_validite(liste_clusters, points)

    #affichage matplotlib
    arbre.affichage_clusters(points=points,
            segments=segments,
            liste_clusters=liste_clusters,
            Npoints=Npoints,
            Nleaf=Nleaf,
            a=a,
            afficher_ids=True
    )

    # On teste les clusters pour les cercles avec le maillage en étoile
    print("Test de validité : clusters cercle vs maillage etoile ")
    points_etoile, *_ = fonctions.maillage_segments(Npoints, a, forme="etoile")
    arbre.affichage_clusters(points=points_etoile,
                segments=segments,
                liste_clusters=liste_clusters,
                Npoints=Npoints,
                Nleaf=Nleaf,
                a=a,
                afficher_ids=True
    )
    arbre.test_validite(liste_clusters, points_etoile)

    
ind_points =[]
points = []
Test_points =[]





