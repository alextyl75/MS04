import fonctions
import arbre_fonctions
import afficher_arbre_fonctions
import test_arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt
import random as rd 
import time 

test_q1_erreur = False
test_q1_temps = True

if test_q1_erreur:
    Npoints = 100
    a = 1
    k = 5
    nq = nq_tilde = 10
    Nleaf = 5

    ETA = 3

    tol_tab = [1e-1, 1e-2, 1e-3, 1e-4]

    # Generation du maillage
    points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)
    A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)

    erreurs_tab = []

    for tol in tol_tab:
        H = arbre_fonctions.H_matrix(A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, tol)

        num_vecteurs = 100
        vecteurs_rd = [
            np.random.randn(Npoints) + 1j * np.random.randn(Npoints) 
            for _ in range(num_vecteurs)
        ]

        erreurs_local = []

        for i, x in enumerate(vecteurs_rd):
            Ax = A_BEM @ x
            Hx = arbre_fonctions.Hmatvec(H, x)
            
            err_rel = np.linalg.norm(Hx - Ax) / np.linalg.norm(Ax)
            erreurs_local.append(err_rel)

        erreurs_tab.append(erreurs_local)

    erreurs_mat = np.array(erreurs_tab)
    erreur_mean = np.mean(erreurs_mat, axis=1)

    # ---------------------------------------------------------
    # Creation de la figure avec espace reserve pour le texte
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Ajuster les marges pour laisser de la place a droite
    plt.subplots_adjust(right=0.72)

    # Courbe d'erreur et ligne de reference
    ax.loglog(tol_tab, erreur_mean, 'o-', color='navy', linewidth=2, label=r"Erreur moyenne $\varepsilon_{mv}$")
    ax.loglog(tol_tab, tol_tab, 'r--', linewidth=1.5, label=r"Référence $y = \mathrm{tol}$")

    ax.set_xlabel("Tolérance ACA (tol)", fontsize=11)
    ax.set_ylabel(r"Erreur relative moyenne $\varepsilon_{mv}(x)$", fontsize=11)
    ax.set_title("Erreur du produit H-matrice-vecteur vs Tolérance ACA", fontsize=12, fontweight='bold')
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    # Inverser l'axe x pour observer la convergence (tol diminue de gauche a droite)
    ax.invert_xaxis()

    # Formattage du texte des parametres du modele
    params_text = (
        "Paramètres du modèle :\n"
        "─────────────────────\n"
        f"• Npoints  = {Npoints}\n"
        f"• Rayon a  = {a}\n"
        f"• Onde k   = {k}\n"
        f"• Nleaf    = {Nleaf}\n"
        f"• ETA (η)  = {ETA}\n"
        f"• N_vec    = {num_vecteurs}"
    )

    # Encadre textuel sur le cote droit de la figure
    fig.text(
        0.75, 0.5, params_text,
        fontsize=10,
        family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.3, edgecolor='gray')
    )

    plt.show()

if test_q1_temps:
    a = 1
    k = 5
    nq = nq_tilde = 10
    Nleaf = 5
    ETA = 3
    tol = 1e-2

    Npoints_tab = [100, 200, 300]
    num_vecteurs = 5

    temps_BEM = []
    temps_H = []

    for Npoints in Npoints_tab:
        # 1. Génération du maillage et assemblage (coût offline)
        points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="cercle")

        # 2. Génération des vecteurs de la bonne taille Npoints
        vecteurs_rd = [
            np.random.randn(Npoints) + 1j * np.random.randn(Npoints)
            for _ in range(num_vecteurs)
        ]

        temps_BEM_local = []
        temps_H_local = []

        # 3. Mesure du temps pour le produit matrice-vecteur (coût online)
        for x in vecteurs_rd:
            # Chronométrage Produit Matrice-Vecteur Dense (Ax)
            t_start = time.perf_counter()
            A_BEM = fonctions.A_assemble(points, segments, nq, nq_tilde, Npoints, k)
            Ax = A_BEM @ x
            t_end = time.perf_counter()
            temps_BEM_local.append(t_end - t_start)

            # Chronométrage Produit Matrice-Vecteur H-matrice (Hx)
            t_start = time.perf_counter()
            arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
            arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)
            H = arbre_fonctions.H_matrix(A_BEM, arbre_bloc, fonctions.ACA_complet_tolerance, tol)

            Hx = arbre_fonctions.Hmatvec(H, x)
            t_end = time.perf_counter()
            temps_H_local.append(t_end - t_start)

        # Calcul des moyennes pour le Npoints courant
        temps_BEM.append(np.mean(temps_BEM_local))
        temps_H.append(np.mean(temps_H_local))

    # ---------------------------------------------------------
    # Création du graphique comparatif (Temps vs Npoints)
    # ---------------------------------------------------------
# ---------------------------------------------------------
    # Création du graphique avec ordonnées en échelle log
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.subplots_adjust(right=0.72)

    # Tracé des courbes
    ax.plot(Npoints_tab, temps_BEM, 's-', color='crimson', linewidth=2, label=r"Dense (BEM) : $\mathcal{O}(N^2)$")
    ax.plot(Npoints_tab, temps_H, 'o-', color='navy', linewidth=2, label=r"$\mathcal{H}$-Matrice : $\mathcal{O}(N \log N)$")

    # ÉCHELLE LOGARITHMIQUE SUR L'AXE Y
    ax.set_yscale('log')

    ax.set_xlabel("Nombre de points ($N$)", fontsize=11)
    ax.set_ylabel("Temps de calcul moyen $A \cdot x$  [log]", fontsize=11)
    ax.set_title("Temps de calcul du produit matrice-vecteur (échelle log)", fontsize=12, fontweight='bold')
    
    # Grille adaptée aux échelles logarithmiques (ticks principaux et secondaires)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    # Encadré latéral des paramètres du modèle
    params_text = (
        "Paramètres du modèle :\n"
        "─────────────────────\n"
        f"• Rayon a  = {a}\n"
        f"• Onde k   = {k}\n"
        f"• Nleaf    = {Nleaf}\n"
        f"• ETA (η)  = {ETA}\n"
        f"• Tol (tol)= {tol}\n"
        f"• N_vec    = {num_vecteurs}"
    )

    fig.text(
        0.75, 0.5, params_text,
        fontsize=10,
        family='monospace',
        verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.3, edgecolor='gray')
    )

    plt.show()




    


        



