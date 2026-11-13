
import os
import time
import csv
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import final
import fonctions

if __name__ == "__main__":

    # Paramètres
    Npoints = 10000
    a = 1
    k = 5
    nq = nq_tilde = 10
    Nleaf = 5
    ETA = 3
    tol = 1e-4

    # Nombre de threads utilisés
    n_workers = os.cpu_count() or 4

    N_tab = [10000, 25000, 50000]

    # Fichier de résultats
    fichier = "resultats_Hmatrice.csv"

    with open(fichier, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")

        writer.writerow([
            "Npoints",
            "Nombre de coefficients",
            "Memoire",
            "Taux de compression",
            "Temps construction",
            "Temps GMRES",
            "Residuel relatif",
            "Info GMRES"
        ])

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

            # 2. Construction de la H-matrice
            t0 = time.perf_counter()

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

            t_par = time.perf_counter() - t0

            print("\nTaille du problème :", Npoints)

            # 3. Calcul de la mémoire et du nombre de coefficients
            mem_H = 0
            taille_memoire = 0

            for cle, bloc in H.items():

                if isinstance(bloc, tuple):
                    U, V = bloc
                    mem_H += U.size + V.size

                else:
                    mem_H += bloc.size

            # Taux de compression
            taux_compression = mem_H / Npoints**2

            print("Nombre de coefficients stockés :", mem_H)
            print(f"Mémoire utilisée : {mem_H:.2f}")
            print(f"Taux de compression : {taux_compression:.4f} %")
            print(f"Temps de construction : {t_par:.4f} s")

            # Résolution sans assemblage de la matrice
            b = final.B(points, segments, Npoints, nq, k)

            # Opérateur linéaire basé directement sur la H-matrice
            A = LinearOperator(
                shape=(Npoints, Npoints),
                matvec=lambda x: final.Hmatvec(H, x),
                dtype=np.result_type(b.dtype, np.float64)
            )

            t0 = time.perf_counter()

            p_num, info = gmres(A, b, rtol=1e-10)

            t_gmres = time.perf_counter() - t0

            # Résidu relatif sans assemblage
            residu = np.linalg.norm(final.Hmatvec(H, p_num) - b) / np.linalg.norm(b)

            print(f"Temps de résolution GMRES : {t_gmres:.4f} s")
            print(f"Résidu relatif : {residu:.4e}")
            print("Info GMRES :", info)

            # 6. Enregistrement des résultats
            writer.writerow([
                Npoints,
                mem_H,
                f"{mem_H:.6f}",
                f"{taux_compression:.6f}",
                f"{t_par:.6f}",
                f"{t_gmres:.6f}",
                f"{residu:.6e}",
                info
            ])

            # Sauvegarde immédiate après chaque taille
            f.flush()

    print(f"\nTous les résultats ont été enregistrés dans : {fichier}")
