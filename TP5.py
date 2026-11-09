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


<<<<<<< HEAD
=======
graphe_q3_rapport = False
>>>>>>> 05e9760171ccb8c41810d64bc72319755825069c

testq5_frequence = False 
testq5_ddl = False
testq5_cst = False

testq5_erreurrelative = True

if testq1:
    # Parametres
    Npoints = 100
    a = 1
    Nleaf = 5

    ETA = 3

    # Generation du maillage
    points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)

    delta = 0.49
    box1 = [(1 + delta,1+ delta),(1+ delta,2+ delta),(2+ delta,1+ delta),(2+ delta,2+ delta)]
    box2 = [(0,0),(0,1),(1,0),(1,1)]
    noeud1 = arbre_fonctions.Noeud(box1, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    noeud2 = arbre_fonctions.Noeud(box2, list_pts = [],niveau=0, indice_local=0,diam = np.sqrt(2))
    print(arbre_fonctions.distance_box(noeud1,noeud2))
    print(arbre_fonctions.is_admissible(noeud1,noeud2,eta = 2.))

if testq2:

    # Parametres
    Npoints = 100
    a = 1
    Nleaf = 5

    ETA = 3

    # Generation du maillage
    points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)

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

if testq5_frequence:

    #Maillage
    a = 1
    k = 5
    Npoints = 200

    points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")

    # Parametres BEM
    nq = nq_tilde = 10
    A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)

    # Parametres H-hierarchique
    Nleaf = 5
    eta = 3.0
    epsilon_aca = 1e-1

    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, eta)

    k_tab = [0.1, 1 , 2.5 , 5 , 10 , 15, 20, 25, 30, 40, 50, 75, 100]
    taux_economie_tab = []

    for k in k_tab:
        A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)
        N_h = arbre_fonctions.H_matrice_gain_memoire(A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, epsilon_aca)

        ratio_stockage = N_h / Npoints**2
        taux_economie = 1 - ratio_stockage

        taux_economie_tab.append(taux_economie)

        # --- CRÉATION DU GRAPHE MATPL
        
    print("ici", taux_economie_tab)
    # Création de la figure et réservation de l'espace à droite pour les paramètres
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.72)

    # Courbe du taux d'économie vs k
    ax.plot(
        k_tab,
        taux_economie_tab,
        marker="o",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        label="Taux d'économie ",
    )

    # Titres et axes
    ax.set_title(
        "Taux d'économie mémoire en fonction du nombre d'onde $k$",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Nombre d'onde $k$ ", fontsize=11)
    ax.set_ylabel("Gain mémoire ", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Encadré des paramètres sur le côté droit
    param_text = (
        "PARAMÈTRES DU TEST\n"
        "------------------\n"
        f"a             : {a}\n"
        f"Npoints       : {Npoints}\n"
        f"Tolérance ACA : {epsilon_aca:.1e}\n"
        f"ETA           : {eta}\n"
        f"Nleaf         : {Nleaf}\n"
    )

    # Placement du bloc de texte à droite de l'axe
    fig.text(
        0.75,
        0.5,
        param_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    plt.show()

            
if testq5_ddl:

    #Maillage
    a = 1
    k = 5
    
    # Parametres BEM
    nq = nq_tilde = 10

    # Parametres H-hierarchique
    Nleaf = 5
    eta = 3.0
    epsilon_aca = 1e-1
    Npoints_tab = [100, 250, 500, 1000]
    taux_economie_tab = []

    for Npoints in Npoints_tab:
        points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
        arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
        arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, eta)
        
        A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)
        N_h = arbre_fonctions.H_matrice_gain_memoire(A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, epsilon_aca)

        ratio_stockage = N_h / Npoints**2
        taux_economie = 1 - ratio_stockage

        taux_economie_tab.append(taux_economie)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.72)  # Réservation de l'espace à droite

    # Courbe du taux d'économie vs Npoints
    ax.plot(
        Npoints_tab,
        taux_economie_tab,
        marker="s",
        color="#2ca02c",
        linewidth=2,
        markersize=7,
        label="Gain mémoire ",
    )

    # Configuration des axes et du titre
    ax.set_title(
        "Taux d'économie mémoire en fonction du nombre de DDL ($N$)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Nombre de degrés de liberté $N$ (Npoints)", fontsize=11)
    ax.set_ylabel("Gain mémoire ", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Encadré des paramètres fixes sur le côté droit
    param_text = (
        "PARAMÈTRES DU TEST\n"
        "------------------\n"
        f"a             : {a}\n"
        f"Nombre d'onde k: {k}\n"
        f"Tolérance ACA : {epsilon_aca:.1e}\n"
        f"ETA           : {eta}\n"
        f"Nleaf         : {Nleaf}\n"
    )

    fig.text(
        0.75,
        0.5,
        param_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    plt.show()

if testq5_cst:
    # Parametres BEM
    ka = 5
    nq = nq_tilde = 10

    # Parametres H-hierarchique
    Nleaf = 5
    eta = 3.0
    epsilon_aca = 1e-1

    Npoints_tab = [100, 250, 500, 750, 1000]
    k_tab = [0.1, 0.5, 1, 5 , 10]
    a_tab = [5/k for k in k_tab]

    taux_economie_tab = []

    for i in range(len(k_tab)):
        Npoints = Npoints_tab[i]
        k = k_tab[i]
        a = a_tab[i]

        points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
        arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
        arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, eta)
        
        A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)
        N_h = arbre_fonctions.H_matrice_gain_memoire(A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, epsilon_aca)

        ratio_stockage = N_h / Npoints**2
        taux_economie = 1 - ratio_stockage

        taux_economie_tab.append(taux_economie)

        fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(right=0.72)  # Marge à droite pour l'encadré des paramètres

    # Courbe du taux d'économie vs Npoints
    ax.plot(
        Npoints_tab,
        taux_economie_tab,
        marker="d",
        color="#d62728",
        linewidth=2,
        markersize=7,
        label="Gain mémoire (%)",
    )

    # Titres et axes
    ax.set_title(
        r"Taux d'économie mémoire à taille électrique constante ($ka = 5$)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Nombre de degrés de liberté $N$ (Npoints)", fontsize=11)
    ax.set_ylabel("Gain mémoire ", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Annotations pour montrer le couple (N, k) sur chaque point
    for i in range(len(Npoints_tab)):
        ax.annotate(
            f"k={k_tab[i]}",
            (Npoints_tab[i], taux_economie_tab[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color="#333333",
        )

    # Encadré des paramètres sur le côté droit
    param_text = (
        "PARAMÈTRES DU TEST\n"
        "------------------\n"
        f"ka            : {ka}\n"
        f"Tolérance ACA : {epsilon_aca:.1e}\n"
        f"ETA           : {eta}\n"
        f"Nleaf         : {Nleaf}\n"
        "------------------\n"
        "Régime : N et k\n"
        "croissants"
    )

    fig.text(
        0.75,
        0.5,
        param_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

if testq5_erreurrelative:
    # --- 1. Paramètres de l'expérience ---
    Npoints = 200
    a = 1
    k = 1
    Nleaf = 5
    nq = nq_tilde = 10
    ETA = 3.0

    # Plage de tolérance étendue de 1e-1 à 1e-5
    tol_tab = [1e-1, 1e-2, 1e-3, 1e-4]
    erreur_tab = []
    taux_tab = []

    # --- 2. Génération du maillage et assemblages initiaux ---
    points, segments, milieux, longueurs, normales = (
        fonctions.maillage_segments(Npoints, a, forme="cercle")
    )
    arbre_cluster = arbre_fonctions.construction_arbre(
        Npoints, Nleaf, points, strat="mediane"
    )
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)
    A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)

    norme_A = np.linalg.norm(A_BEM, ord="fro")

    # --- 3. Boucle de calcul sur les tolérances ACA ---
    for tol in tol_tab:
        H = arbre_fonctions.H_matrix(
            A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, tol
        )
        N_h = arbre_fonctions.H_matrice_gain_memoire(
            A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, tol
        )
        H_assemble = arbre_fonctions.H_matrix_assemblage(H, Npoints)

        # Gain mémoire exprimé en fraction (entre 0 et 1)
        ratio_stockage = N_h / (Npoints**2)
        taux_economie_fraction = 1.0 - ratio_stockage

        # Erreur relative absolue en norme de Frobenius
        erreur_rel = np.linalg.norm(A_BEM - H_assemble, ord="fro") / norme_A

        erreur_tab.append(erreur_rel)
        taux_tab.append(taux_economie_fraction)

    # --- 4. Génération de la figure (2 subplots 1x2 + Encadré paramètres) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Réservation de la marge à droite pour les paramètres (right=0.78)
    fig.subplots_adjust(left=0.08, right=0.78, wspace=0.32)

    color_err = "#1f77b4"
    color_gain = "#d62728"

    # --- Subplot 1 : Erreur relative uniquement (Échelle log-log) ---
    ax1.loglog(
        tol_tab,
        erreur_tab,
        marker="o",
        color=color_err,
        linewidth=2,
        label=r"Erreur $\|A-A_H\|_F / \|A\|_F$",
    )
    ax1.set_xlabel(r"Tolérance ACA ($\varepsilon_{aca}$)", fontsize=11)
    ax1.set_ylabel(r"Erreur relative $\varepsilon_H$", fontsize=11)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.set_title(r"1. Erreur relative $\varepsilon_H$", fontweight="bold")
    ax1.legend(loc="best")

    # --- Subplot 2 : Gain mémoire uniquement (Fixé entre 0 et 1) ---
    ax2.semilogx(
        tol_tab,
        taux_tab,
        marker="s",
        color=color_gain,
        linewidth=2,
        label="Gain mémoire (fraction)",
    )
    ax2.set_xlabel(r"Tolérance ACA ($\varepsilon_{aca}$)", fontsize=11)
    ax2.set_ylabel("Gain mémoire (ratio)", fontsize=11)
    ax2.set_ylim(0.0, 1.0)  # Axe Y fixé strictement entre 0 et 1
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.set_title("2. Gain mémoire (Ratio)", fontweight="bold")
    ax2.legend(loc="best")

    # --- 5. Encadré des paramètres sur le côté droit ---
    param_text = (
        "PARAMÈTRES DU TEST\n"
        "------------------\n"
        f"Geométrie : Cercle\n"
        f"a         : {a}\n"
        f"Npoints   : {Npoints}\n"
        f"Nombre k  : {k}\n"
        f"ETA       : {ETA}\n"
        f"Nleaf     : {Nleaf}\n"
        "------------------\n"
        r"Tol. ACA  : 1e-1 $\to$ 1e-5"
    )

    fig.text(
        0.81,
        0.5,
        param_text,
        transform=fig.transFigure,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#cccccc"),
    )

    plt.suptitle(
        r"Évaluation du compromis entre Précision ($\varepsilon_H$) et Économie Mémoire ($N_{points}=500$)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.show()





                

        