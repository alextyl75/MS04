import fonctions
import arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

testq1 = False
testq2 = False
testq3 = True

# Parametres
Npoints = 200
a = 1
Nleaf = 5

ETA = 3

# Generation du maillage
points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points)

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
    arbre_fonctions.afficher_arbre_bloc(arbre_bloc[0])
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
    Npoints = 100
    Nleaf = 5
    eta = 2.0

    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="etoile")
    A = fonctions.A_assemble(points=points, segments=segments, nq=5, nq_tilde=5, N=Npoints, k=k)

    # 1. Construction de l'arbre de clusters et de l'arbre de blocs
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, eta)

    # 2. Fonction de visualisation de la H-matrice
    def visualise_matrice(arbre_bloc, arbre_cluster, Npoints, eta):
        # Récupération de la permutation (Q4) pour que chaque cluster ait des indices consécutifs
        perm = arbre_fonctions.permutation_indice(arbre_cluster)
        perm_inv = {ancien_id: nouvel_id for nouvel_id, ancien_id in enumerate(perm)}

        fig, ax = plt.subplots(figsize=(8, 8))

        nb_admissibles = 0
        nb_denses = 0

        for noeud_bloc in arbre_bloc:
            # On ne trace que les feuilles de l'arbre de blocs
            if noeud_bloc.type_feuille in ("feuille_admissible", "feuille_dense"):
                
                # Nouveaux indices (consécutifs) pour les lignes (tau) et colonnes (sigma)
                idx_tau = [perm_inv[pt[0]] for pt in noeud_bloc.tau.list_pts]
                idx_sigma = [perm_inv[pt[0]] for pt in noeud_bloc.sigma.list_pts]

                i_min, i_max = min(idx_tau), max(idx_tau)
                j_min, j_max = min(idx_sigma), max(idx_sigma)

                # Coordonnées et dimensions du bloc dans la matrice
                # En représentation matricielle : x = colonne (j), y = ligne (i)
                largeur = j_max - j_min + 1
                hauteur = i_max - i_min + 1

                if noeud_bloc.type_feuille == "feuille_admissible":
                    couleur_fond = "#4ade80"   # Vert
                    nb_admissibles += 1
                elif noeud_bloc.type_feuille == "feuille_dense":
                    couleur_fond = "#f87171"   # Rouge
                    nb_denses += 1

                # Ajout du rectangle (avec bordure noire pour bien voir le découpage hiérarchique)
                rect = patches.Rectangle(
                    (j_min, i_min), largeur, hauteur,
                    linewidth=0.6, edgecolor="black", facecolor=couleur_fond, alpha=0.85
                )
                ax.add_patch(rect)

        # Réglage des axes au format matriciel (origine (0,0) en haut à gauche)
        ax.set_xlim(0, Npoints)
        ax.set_ylim(Npoints, 0)  # Inversion de l'axe Y pour avoir la ligne 0 en haut
        ax.set_aspect('equal')
        ax.xaxis.tick_top()      # Indices des colonnes en haut comme dans une matrice
        ax.xaxis.set_label_position('top')
        ax.set_xlabel("Indices des colonnes ($\sigma$ renuméroté)")
        ax.set_ylabel("Indices des lignes ($\\tau$ renuméroté)")
        ax.set_title(
            f"Structure des blocs de la $\mathcal{{H}}$-matrice ($\eta = {eta}$)\n"
            f"{nb_admissibles} blocs admissibles (verts) | {nb_denses} blocs denses (rouges)",
            pad=20
        )

        # Légende
        legend_elements = [
            patches.Patch(facecolor="#4ade80", edgecolor="black", label="Feuille admissible (compressible)"),
            patches.Patch(facecolor="#f87171", edgecolor="black", label="Feuille dense (non admissible)")
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plt.show()

    # Appel de la fonction
    visualise_matrice(arbre_bloc, arbre_cluster, Npoints, eta)

    
            

    