import fonctions
import arbre_fonctions
import afficher_arbre_fonctions
import test_arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt
import random as rd 

test_q1 = True

if test_q1:
    Npoints = 100
    a = 1
    k = 5
    nq = nq_tilde = 5
    Nleaf = 5
    tol = 1e-1

    ETA = 3

    # Generation du maillage
    points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)
    A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)

    H = arbre_fonctions.H_matrix(A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, tol)

    vecteur_rd = np.array([rd.random() for i in range(Npoints)])

    # Produit matrice-vecteur exact
    Ax = A_BEM @ vecteur_rd

    # Produit matrice-vecteur H-matrice
    Hx = arbre_fonctions.Hmatvec(H, vecteur_rd)

    # Erreur relative en norme 2 (norme euclidienne)
    erreur = np.linalg.norm(Ax - Hx) / np.linalg.norm(Ax)

    print(erreur)
        



