
import os
import time
import numpy as np
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

    # Nombre de threads utilisés
    n_workers = os.cpu_count() or 4

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

    print(f"Test pour Npoints = {Npoints}")
    print(f"Nombre de cœurs disponibles : {os.cpu_count()}")
    print(f"Nombre de threads utilisés : {n_workers}\n")

    # 3. Version parallèle avec threads
    print("Construction parallèle...")

    t0 = time.perf_counter()

    H_par = final.H_matrix_light(
        arbre_bloc=arbre_bloc,
        aca_function=final.ACA_partiel_free_bloc,
        tol=1e-4,
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

    print(f"Temps parallèle : {t_par:.4f} s")


    # 2. Version séquentielle
    print("Construction séquentielle...")

    t0 = time.perf_counter()

    H_seq = final.H_matrix_light_old(
        arbre_bloc=arbre_bloc,
        aca_function=final.ACA_partiel_free_bloc,
        tol=1e-4,
        points=points,
        segments=segments,
        k=k,
        x_quad=x_quad,
        w=w,
        x_quad_tilde=x_quad_tilde,
        w_tilde=w_tilde,
        fonctions=final.G,
    )

    t_seq = time.perf_counter() - t0

    print(f"Temps séquentiel : {t_seq:.4f} s\n")

    print(f"Accélération : {t_seq / t_par:.2f}x\n")

    # 4. Vérification numérique
    np.random.seed(42)

    x = np.random.rand(Npoints) + 1j * np.random.rand(Npoints)

    y_seq = final.Hmatvec(H_seq, x)
    y_par = final.Hmatvec(H_par, x)

    err = np.linalg.norm(y_seq - y_par) / np.linalg.norm(y_seq)

    print(f"Différence relative : {err:.2e}")

    if err < 1e-10:
        print("Vérification numérique réussie.")
    else:
        print("Attention : les deux résultats sont différents.")
