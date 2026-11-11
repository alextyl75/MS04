import fonctions
import arbre_fonctions
import afficher_arbre_fonctions
import test_arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt
import random as rd 

test_q1 = False
test_q5 = True

if test_q1:
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
        





if test_q5:
    Npoints = 400
    a = 1
    k = 5
    nq = nq_tilde = 10
    Nleaf = 5

    ETA = 3
    # Generation du maillage
    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    arbre_cluster = arbre_fonctions.construction_arbre(Npoints, Nleaf, points, strat="mediane")
    arbre_bloc = arbre_fonctions.arbre_bloc_construction(arbre_cluster, ETA)

    x_quad, w = np.polynomial.legendre.leggauss(nq)
    x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)

    print("Construction de la H-Matrice...")
    H = arbre_fonctions.H_matrix_light(
        arbre_bloc=arbre_bloc,
        aca_function=arbre_fonctions.ACA_partiel_free_bloc,
        tol=1e-4,
        points=points,
        segments=segments,
        k=k,
        x_quad=x_quad,
        w=w,
        x_quad_tilde=x_quad_tilde,
        w_tilde=w_tilde,
        fonctions=fonctions.G,  # ou fonctions selon votre implémentation
    )

    # ==========================================
    # TESTS DE VALIDATION (Précision et Mémoire)
    # ==========================================
    print("\n--- Début du test de validation ---")
    
    # 1. Vecteur complexe aléatoire
    np.random.seed(42)
    x = np.random.rand(Npoints) + 1j * np.random.rand(Npoints)

    # 2. Produit Matrice-Vecteur H-Matrice
    y_H = arbre_fonctions.Hmatvec(H, x)

    # 3. Assemblage de la matrice PLEINE globale pour référence
    indices_full = np.ix_(np.arange(Npoints), np.arange(Npoints))
    A_full = arbre_fonctions.A_assemble_bloc(
        points, segments, indices_full, 
        x_quad, w, x_quad_tilde, w_tilde, 
        k, fonctions=fonctions.G
    )

    # 4. Produit Matrice-Vecteur classique
    y_full = A_full @ x

    # 5. Calcul de l'erreur relative globale
    erreur_relative = np.linalg.norm(y_H - y_full) / np.linalg.norm(y_full)
    print(f"Erreur relative ||y_H - y_full|| / ||y_full|| : {erreur_relative:.2e}")
    
    # Le test passe si l'erreur est globalement du même ordre de grandeur que 'tol'
    if erreur_relative < 1e-3:
        print("✅ Test de précision validé !")
    else:
        print("❌ Problème de précision, l'erreur est trop élevée.")

    # 6. Calcul du taux de compression mémoire
    mem_full = Npoints * Npoints
    mem_H = 0
    nb_admissibles = 0
    nb_denses = 0

    for cle, bloc in H.items():
        if isinstance(bloc, tuple):
            U, V = bloc
            mem_H += U.size + V.size
            nb_admissibles += 1
        else:
            mem_H += bloc.size
            nb_denses += 1

    taux_compression = (1 - mem_H / mem_full) * 100
    print(f"\n--- Statistiques H-Matrice ---")
    print(f"Blocs admissibles (rang faible) : {nb_admissibles}")
    print(f"Blocs denses (exacts)           : {nb_denses}")
    print(f"Éléments stockés (H-Matrice)    : {mem_H}")
    print(f"Éléments stockés (Matrice N x N): {mem_full}")
    print(f"Taux de compression mémoire     : {taux_compression:.2f}%")



