import fonctions
import arbre_fonctions
import afficher_arbre_fonctions
import test_arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

testq1 = False
testq2 = False
testq3 = False
graphe_q3_rapport = False
testq4 = True



# Parametres
Npoints = 100
a = 1
Nleaf = 5

ETA = 3

# Generation du maillage
points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)

if testq1:
    delta = 0.49
    box1 = [(1 + delta,1+ delta),(1+ delta,2+ delta),(2+ delta,1+ delta),(2+ delta,2+ delta)]
    box2 = [(0,0),(0,1),(1,0),(1,1)]
    noeud1 = arbre_fonctions.Noeud(box1, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    noeud2 = arbre_fonctions.Noeud(box2, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    print(arbre_fonctions.distance_box(noeud1,noeud2))
    print(arbre_fonctions.is_admissible(noeud1,noeud2,eta = 2.))

if testq2:
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)
    afficher_arbre_fonctions.afficher_arbre_bloc(arbre_bloc[0])
    arbre_fonctions.arbre_bloc_partition(arbre_bloc[0], Npoints)

# if testq3:
#     a = 1
#     k = 5
#     Npoints = 200
#     points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
#     A = fonctions.A_assemble(points=points, segments=segments, nq=5, nq_tilde=5, N=Npoints, k=k)
#     def visualise_matrice(arbre_bloc):
#         for noeud_bloc in arbre_bloc:
#             if noeud_bloc.type_feuille == "feuille admissible":
#                 "mettre le bloc correspond à A_sigma,tau en vert"
#             if noeud_bloc.type_feuille == "feuille_dense":
#                 "mettre le bloc correspond à A_sigma,tau en rouge"
#         "visualiser la matrice A avec ses blocs hierarchiques en rouge et vert"

if testq3:
    a = 1
    k = 5
    Npoints = 200
    Nleaf = 5
    eta = 5.0

    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="etoile")
    A = fonctions.A_assemble(points=points, segments=segments, nq=5, nq_tilde=5, N=Npoints, k=k)
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, eta)

    afficher_arbre_fonctions.visualise_H_matrice(arbre_bloc, arbre_cluster, Npoints, eta)

if graphe_q3_rapport:
    a = 1
    k = 5
    Npoints = 1000
    Nleaf = 5
    eta = 3.0

    points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")

        # for noeud_bloc in arbre_bloc:
        #     # On ne trace que les feuilles de l'arbre de blocs
        #     if noeud_bloc.type_feuille in ("feuille_admissible", "feuille_dense"):
                
        #         # Nouveaux indices (consécutifs) pour les lignes (tau) et colonnes (sigma)
        #         print(noeud_bloc.tau.list_pts[0],noeud_bloc.tau.list_pts[0][0])
        #         idx_tau = [perm_inv[pt[0]] for pt in noeud_bloc.tau.list_pts]
        #         idx_sigma = [perm_inv[pt[0]] for pt in noeud_bloc.sigma.list_pts]
    # 1. Vos 6 valeurs de eta à tester
    eta_values = [0.2, 0.5, 1.0, 2, 3.0, 4.0]

    afficher_arbre_fonctions.visualise_H_matrice_6_eta(
        creer_arbre_bloc_func=lambda e: arbre_fonctions.arbre_bloc_construction(arbre_cluster, e),
        arbre_cluster=arbre_cluster,
        Npoints=Npoints,
        eta_list=eta_values
    )

######Question 4

if testq4:
    a = 1
    k = 5
    Npoints = 700
    Nleaf = 15
    eta = 3.0

    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="etoile")
    A_bem = fonctions.A_assemble(points=points, segments=segments, nq=5, nq_tilde=5, N=Npoints, k=k)
    print("pass")
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, eta)
    
    afficher_arbre_fonctions.visualise_H_matrice(arbre_bloc, arbre_cluster, Npoints, eta)
        # Exemple 1 : Comparaison stricte à taille égale (autour de la taille des feuilles Nleaf)
    # Idéal pour répondre à "Comparer avec des blocs non admissibles de tailles comparables" (4b)
    afficher_arbre_fonctions.analyser_question_4(
        A=A_bem,                  # Votre matrice BEM pleine (NxN)
        arbre_bloc=arbre_bloc,    # Votre liste renvoyée par arbre_bloc_construction
        eps=1e-6,                 # Tolérance affichée sur le graphique
        liste_eps_tableau=[1e-3, 1e-6, 1e-9],
        nb_admissibles=3,
        nb_denses=3,
        taille_min=10,
        taille_max=40,            # Force à ne regarder que des blocs de taille similaire aux feuilles denses
        forcer_meme_taille=True
    )

    # Exemple 2 : Voir le rang sur de très grands blocs admissibles (ex: 60x60 à 250x250)
    # Idéal pour montrer en 4(c) que même un bloc 250x250 a un rang r_eps ~ 5 à 10 !
    afficher_arbre_fonctions.analyser_question_4(
        A=A_bem,
        arbre_bloc=arbre_bloc,
        eps=1e-6,
        nb_admissibles=4,
        nb_denses=2,
        taille_min=30,
        taille_max=300,
        forcer_meme_taille=False
    )
            

    