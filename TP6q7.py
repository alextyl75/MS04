import os
import time
import numpy as np
import final
import fonctions
import matplotlib.pyplot as plt

test_k = False
test_N = False
test_kN = True

if __name__ == "__main__":
    # Nombre de threads utilisés
    n_workers = os.cpu_count() or 4

    if test_k:
        # Paramètres
        Npoints = 5000
        a = 1
        nq = nq_tilde = 10
        Nleaf = 5
        ETA = 3
        tol=1e-4

        k_tab = [1, 5, 10, 20]
        memoire_tab = []

        # 1. Maillage et arbres
        points, segments, milieux, longueurs, normales = fonctions.maillage_segments(
            Npoints, a, forme="cercle"
        )
        segments = np.asarray(segments, dtype=np.int64)
    
        arbre_cluster = final.construction_arbre(
            Npoints, Nleaf, points, strat="mediane"
        )
        arbre_bloc = final.arbre_bloc_construction(arbre_cluster, ETA)
        x_quad, w = np.polynomial.legendre.leggauss(nq)
        x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)
    
        for k in k_tab:

            H = final.H_matrix_light(
            arbre_bloc=arbre_bloc,
            aca_function=final.ACA_partiel_free_bloc,
            tol=tol,
            points=points,
            segments=segments,
            k=k,
            x_quad=x_quad,
            w=w,
            x_quad_tilde=x_quad_tilde,
            w_tilde=w_tilde,
            fonctions=final.G,
            n_workers=n_workers,
            seuil_points=0,
            )

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

            memoire_tab.append(taux_compression)

        print(nb_admissibles)
        print(nb_denses)
        print(memoire_tab)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        # Tracé de la mémoire H-matrice et de la référence dense
        ax.plot(k_tab, memoire_tab, "o-", linewidth=2, markersize=6, color="#1f77b4")

        # Titres et axes
        ax.set_title(r"Évolution du gain en mémoire de la $\mathcal{H}$-matrice en fonction de $k$", fontsize=12, pad=12)
        ax.set_xlabel(r"Nombre d'onde $k$", fontsize=11)
        ax.set_ylabel("Gain en mémoire", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper left", frameon=True)

        # Encadré des paramètres (sans nq_tilde)
        params_text = (
        f"Paramètres :\n"
        f"• Npoints = {Npoints}\n"
        f"• a = {a}\n"
        f"• nq = {nq}\n"
        f"• Nleaf = {Nleaf}\n"
        f"• ETA = {ETA}\n"
        f"• tol = {tol}"
        )

        ax.text(
        0.02, 0.78, params_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95)
        )

        plt.tight_layout()
        plt.savefig("memoire_vs_k.png", dpi=150)
        plt.show()

    if test_N:
        # Paramètres
        k = 5
        a = 1
        nq = nq_tilde = 10
        Nleaf = 5
        ETA = 3
        tol=1e-4

        N_tab = [1000, 2000, 4000, 6000, 8000]
        memoire_tab = []
    
        for Npoints in N_tab:

            # 1. Maillage et arbres
            points, segments, milieux, longueurs, normales = fonctions.maillage_segments(
                Npoints, a, forme="cercle"
            )
            segments = np.asarray(segments, dtype=np.int64)
        
            arbre_cluster = final.construction_arbre(
                Npoints, Nleaf, points, strat="mediane"
            )
            arbre_bloc = final.arbre_bloc_construction(arbre_cluster, ETA)
            x_quad, w = np.polynomial.legendre.leggauss(nq)
            x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)


            H = final.H_matrix_light(
            arbre_bloc=arbre_bloc,
            aca_function=final.ACA_partiel_free_bloc,
            tol=tol,
            points=points,
            segments=segments,
            k=k,
            x_quad=x_quad,
            w=w,
            x_quad_tilde=x_quad_tilde,
            w_tilde=w_tilde,
            fonctions=final.G,
            n_workers=n_workers,
            seuil_points=0,
            )

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

            memoire_tab.append(taux_compression)

        print(nb_admissibles)
        print(nb_denses)
        print(memoire_tab)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        # Tracé de la mémoire H-matrice et de la référence dense
        ax.plot(N_tab, memoire_tab, "o-", linewidth=2, markersize=6, color="#1f77b4")

        # Titres et axes
        ax.set_title(r"Évolution du gain en mémoire de la $\mathcal{H}$-matrice en fonction de $Npoints$", fontsize=12, pad=12)
        ax.set_xlabel(r"Nombre de points", fontsize=11)
        ax.set_ylabel("Gain en mémoire", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper left", frameon=True)

        # Encadré des paramètres (sans nq_tilde)
        params_text = (
        f"Paramètres :\n"
        f"• k = {k}\n"
        f"• a = {a}\n"
        f"• nq = {nq}\n"
        f"• Nleaf = {Nleaf}\n"
        f"• ETA = {ETA}\n"
        f"• tol = {tol}"
        )

        ax.text(
        0.02, 0.78, params_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95)
        )

        plt.tight_layout()
        plt.savefig("memoire_vs_k.png", dpi=150)
        plt.show()

if test_kN:
    # Paramètres
    ka = 5
    nq = nq_tilde = 10
    Nleaf = 5
    ETA = 3
    tol = 1e-4

    N_tab = [1000, 2000, 3000, 4000]
    k_tab = [1, 5, 10, 15]
    a_tab = [ka/k for k in k_tab]

    memoire_tab = np.zeros((len(N_tab), len(k_tab)))

    for i, Npoints in enumerate(N_tab):

        for j, k in enumerate(k_tab):

            a = a_tab[j]

            # 1. Maillage et arbres
            points, segments, milieux, longueurs, normales = fonctions.maillage_segments(
                Npoints, a, forme="cercle"
            )

            segments = np.asarray(segments, dtype=np.int64)

            arbre_cluster = final.construction_arbre(
                Npoints, Nleaf, points, strat="mediane"
            )

            arbre_bloc = final.arbre_bloc_construction(arbre_cluster, ETA)

            x_quad, w = np.polynomial.legendre.leggauss(nq)
            x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)

            H = final.H_matrix_light(
                arbre_bloc=arbre_bloc,
                aca_function=final.ACA_partiel_free_bloc,
                tol=tol,
                points=points,
                segments=segments,
                k=k,
                x_quad=x_quad,
                w=w,
                x_quad_tilde=x_quad_tilde,
                w_tilde=w_tilde,
                fonctions=final.G,
                n_workers=n_workers,
                seuil_points=0,
            )

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

            memoire_tab[i, j] = taux_compression

            print(
                f"N = {Npoints}, k = {k}, "
                f"gain mémoire = {taux_compression:.2f} %, "
                f"blocs admissibles = {nb_admissibles}, "
                f"blocs denses = {nb_denses}"
            )

    print("\nMatrice des gains mémoire :")
    print(memoire_tab)

    # Graphe : gain mémoire en fonction de N

    fig, ax = plt.subplots(figsize=(10, 6))

    for j, k in enumerate(k_tab):

        gains = memoire_tab[:, j]

        ax.plot(
            N_tab,
            gains,
            marker="o",
            markersize=7,
            linewidth=2,
            label=fr"$k={k}$"
        )

        # Affichage de k à côté de chaque point
        for i, N in enumerate(N_tab):

            ax.annotate(
                fr"$k={k}$",
                (N, gains[i]),
                textcoords="offset points",
                xytext=(5, 8),
                fontsize=8
            )

    # Ligne de référence : gain nul
    ax.axhline(
        0,
        color="black",
        linewidth=1,
        linestyle="--",
        alpha=0.7
    )

    ax.set_xlabel(r"Nombre de points $N$", fontsize=12)
    ax.set_ylabel("Gain mémoire (%)", fontsize=12)

    ax.set_title(
        r"Évolution du gain mémoire en fonction de $N$"
        "\n"
        r"Comparaison pour différentes valeurs de $k$ ($ka=5$)",
        fontsize=13,
        pad=12
    )

    ax.set_xticks(N_tab)

    ax.grid(
        True,
        linestyle=":",
        alpha=0.7
    )

    ax.legend(
        title="Nombre d'onde",
        loc="best",
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("gain_memoire_vs_N.png", dpi=300)
    plt.show()



