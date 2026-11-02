import fonctions
import arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt

testq1 = True

if testq1:
    delta = 0.49
    box1 = [(1 + delta,1+ delta),(1+ delta,2+ delta),(2+ delta,1+ delta),(2+ delta,2+ delta)]
    box2 = [(0,0),(0,1),(1,0),(1,1)]
    noeud1 = arbre_fonctions.Noeud(box1, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    noeud2 = arbre_fonctions.Noeud(box2, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    print(arbre_fonctions.distance_box(noeud1,noeud2))
    print(arbre_fonctions.is_admissible(noeud1,noeud2,eta = 2.))

if testq3:
    a = 1
    k = 5
    Npoints = 200
    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    A = fonctions.A_assemble(points=points, segments=segments, nq=5, nq_tilde=5, N=Npoints, k=k)
    def visualise_matrice(arbre_bloc):
        for noeud_bloc in arbre_bloc:
            if noeud_bloc.type_feuille == "feuille admissible":
                "mettre le bloc correspond à A_sigma,tau en vert"
            if noeud_bloc.type_feuille == "feuille_dense":
                "mettre le bloc correspond à A_sigma,tau en rouge"
        "visualiser la matrice A avec ses blocs hierarchiques en rouge et vert"

    
            


