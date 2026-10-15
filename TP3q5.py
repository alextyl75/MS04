import time
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 1. FONCTIONS DE FACTORISATION DE RANG FAIBLE
# ==============================================================================

def svd_tronquee(U, sigma, Vt, r):
    Ur = U[:, :r]
    Sigmar = np.diag(sigma[:r])
    Vtr = Vt[:r, :]
    return Ur @ Sigmar @ Vtr


def ACA_iteration(R, rang_iteration, tol_pivot=1e-12):
    """Fait l'itération de ACA et renvoie u_{k+1} et v_{k+1}^T."""
    m, n = R.shape
    i_star, j_star = np.unravel_index(np.argmax(np.abs(R)), R.shape)
    pivot = R[i_star, j_star]

    if np.abs(pivot) < tol_pivot:
        return None, None

    u_k = R[:, j_star] / pivot
    v_k_trans = R[i_star, :]
    return u_k, v_k_trans


def ACA_complet(A, rang_final, epsilon=None):
    """ACA avec pivotage complet."""
    n, m = A.shape
    A_approx = np.zeros((n, m))
    R = A.copy().astype(float)
    rang_obtenu = 0

    for rang_iteration in range(1, rang_final + 1):
        u, v = ACA_iteration(R, rang_iteration)
        if u is None:
            break

        update = np.outer(u, v)
        A_approx = A_approx + update
        R = R - update
        rang_obtenu += 1

        if epsilon is not None:
            norm_prod_uv = np.linalg.norm(u) * np.linalg.norm(v)
            if norm_prod_uv < epsilon * np.linalg.norm(A_approx, "fro"):
                break

    return A_approx, R, rang_obtenu


def ACA_partiel(A, N, epsilon, verbose=False):
    """
    ACA avec pivotage partiel (renvoie les listes u_k_used, v_k_used).
    Complexité O(r^2 * N).
    """
    A_approx_norm = 0.0
    next_indice = 0
    index_used = []
    cols_used = []
    v_k_used = []
    u_k_used = []
    norm_prod_uv = 0.0

    def ACA_partiel_step(index_ligne):
        res_row = A[index_ligne, :].copy().astype(float)
        for u, v in zip(u_k_used, v_k_used):
            res_row -= u[index_ligne] * v

        abs_res_row = np.abs(res_row)
        abs_res_row[cols_used] = -1
        j = np.argmax(abs_res_row)
        pivot = res_row[j]

        if np.abs(pivot) < 1e-12:
            u_k = np.zeros(N)
            v_k = np.zeros(N)
            next_indice_val = (index_ligne + 1) % N
        else:
            res_col = A[:, j].copy().astype(float)
            for u, v in zip(u_k_used, v_k_used):
                res_col -= u * v[j]

            u_k = res_col / pivot
            v_k = res_row
            cols_used.append(j)
            index_used.append(index_ligne)

            col_abs = np.abs(u_k).copy()
            col_abs[index_used] = -1
            next_indice_val = np.argmax(col_abs)

        v_k_used.append(v_k)
        u_k_used.append(u_k)
        return next_indice_val

    step = 1
    while (norm_prod_uv >= epsilon * A_approx_norm) and (step <= N):
        next_indice = ACA_partiel_step(next_indice)

        u_k = u_k_used[-1]
        v_k = v_k_used[-1]
        norm_prod_uv = np.linalg.norm(u_k) * np.linalg.norm(v_k)

        if norm_prod_uv == 0.0:
            u_k_used.pop()
            v_k_used.pop()
            break

        sum_norm = 0.0
        for l in range(len(u_k_used) - 1):
            sum_norm += np.inner(u_k_used[l], u_k) * np.inner(v_k_used[l], v_k)

        valeur_sous_racine = A_approx_norm**2 + 2 * sum_norm + norm_prod_uv**2
        A_approx_norm = np.sqrt(max(0.0, valeur_sous_racine))
        step += 1

    if verbose:
        print(f"Convergence atteinte en {len(u_k_used)} étapes : critère {norm_prod_uv:.2e} < {epsilon * A_approx_norm:.2e}")
    return u_k_used, v_k_used


def reconstruire_ACA_partiel(u_k_used, v_k_used, N=None):
    """Reconstruit la matrice dense à partir des facteurs."""
    if len(u_k_used) == 0:
        if N is None:
            raise ValueError("Préciser N si les listes sont vides.")
        return np.zeros((N, N))

    n_rows, n_cols = len(u_k_used[0]), len(v_k_used[0])
    A_approx = np.zeros((n_rows, n_cols))
    for u, v in zip(u_k_used, v_k_used):
        A_approx += np.outer(u, v)
    return A_approx


# ==============================================================================
# 2. GÉNÉRATEURS DE MATRICES
# ==============================================================================

def matrice_rang_exact_polynome(N, p=5):
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    return (1.0 + np.outer(x, y)) ** p

def matrice_rang_exact_facteurs(N, r_exact=8, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, r_exact))
    Y = rng.standard_normal((N, r_exact))
    return X @ Y.T

def matrice_rang_numerique_coulomb(N):
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(2.0, 3.0, N)
    return 1.0 / np.abs(x[:, None] - y[None, :])

def matrice_rang_numerique_spectre_controle(N, base=10.0, seed=42):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((N, N)))
    V, _ = np.linalg.qr(rng.standard_normal((N, N)))
    sigma = base ** (-np.arange(N, dtype=float))
    return (U * sigma) @ V.T


# ==============================================================================
# 3. GÉNÉRATION DES 4 GRAPHIQUES (SUR DES PAGES/FIGURES SÉPARÉES)
# ==============================================================================

if __name__ == "__main__":
    # Paramètres d'affichage globaux
    plt.rcParams.update({"font.size": 11})
    N_test = 150
    
    # Génération des matrices de test
    A_exact_poly = matrice_rang_exact_polynome(N_test, p=5)         # Rang exact = 6
    A_exact_rand = matrice_rang_exact_facteurs(N_test, r_exact=8)   # Rang exact = 8
    A_num_coulomb = matrice_rang_numerique_coulomb(N_test)          # Rang numérique
    A_num_spectre = matrice_rang_numerique_spectre_controle(N_test) # Rang numérique

    # Pré-calculs SVD pour réutilisation
    _, s_poly, _ = np.linalg.svd(A_exact_poly)
    _, s_rand, _ = np.linalg.svd(A_exact_rand)
    U_c, s_c, Vt_c = np.linalg.svd(A_num_coulomb, full_matrices=False)
    _, s_spec, _ = np.linalg.svd(A_num_spectre)


    # --------------------------------------------------------------------------
    # FIGURE 1 : Spectres des valeurs singulières
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    k_max = 20
    plt.semilogy(range(1, k_max + 1), s_poly[:k_max] / s_poly[0], "o-", label="Polynôme (Rang exact = 6)")
    plt.semilogy(range(1, k_max + 1), s_rand[:k_max] / s_rand[0], "s-", label="Produit XY^T (Rang exact = 8)")
    plt.semilogy(range(1, k_max + 1), s_c[:k_max] / s_c[0], "^-", label="Noyau 1/|x-y| (Rang numérique)")
    plt.semilogy(range(1, k_max + 1), s_spec[:k_max] / s_spec[0], "d--", label="Spectre 10^{-k} (Rang numérique)")
    
    plt.axhline(1e-6, color="red", linestyle=":", label="Tolérance $\\epsilon = 10^{-6}$")
    plt.title("1. Décroissance des valeurs singulières normalisées ($\\sigma_k / \\sigma_1$)")
    plt.xlabel("Indice $k$")
    plt.ylabel("$\\sigma_k / \\sigma_1$ (échelle log)")
    plt.ylim(1e-17, 10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # FIGURE 2 : Erreur relative en fonction du rang
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    norm_A_coul = np.linalg.norm(A_num_coulomb, "fro")
    rangs_test = range(1, 13)
    err_svd, err_aca_c = [], []

    for r in rangs_test:
        A_svd = svd_tronquee(U_c, s_c, Vt_c, r)
        err_svd.append(np.linalg.norm(A_num_coulomb - A_svd, "fro") / norm_A_coul)

        A_acac, _, _ = ACA_complet(A_num_coulomb, rang_final=r)
        err_aca_c.append(np.linalg.norm(A_num_coulomb - A_acac, "fro") / norm_A_coul)

    eps_grid = np.logspace(-1, -13, 13)
    rangs_acap_pts, err_acap_pts = [], []
    for eps in eps_grid:
        u_list, v_list = ACA_partiel(A_num_coulomb, N_test, eps)
        A_acap = reconstruire_ACA_partiel(u_list, v_list, N=N_test)
        rangs_acap_pts.append(len(u_list))
        err_acap_pts.append(np.linalg.norm(A_num_coulomb - A_acap, "fro") / norm_A_coul)

    plt.semilogy(rangs_test, err_svd, "o-", label="SVD Tronquée (optimale)")
    plt.semilogy(rangs_test, err_aca_c, "s--", label="ACA Complet")
    plt.semilogy(rangs_acap_pts, err_acap_pts, "^:", label="ACA Partiel (reconstruit)")
    
    plt.title("2. Convergence de l'erreur relative sur le noyau $1/|x_i - y_j|$")
    plt.xlabel("Rang de l'approximation $r$")
    plt.ylabel("Erreur relative $\\|A - A_r\\|_F / \\|A\\|_F$")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # FIGURE 3 : Rangs obtenus en fonction de la tolérance
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    tolerances = np.logspace(-1, -12, 12)
    norm_exact = np.linalg.norm(A_exact_rand, "fro")
    
    r_svd_exact, r_acac_exact, r_acap_exact = [], [], []
    r_svd_num, r_acac_num, r_acap_num = [], [], []

    for eps in tolerances:
        # Cas 1 : Rang exact (r=8)
        res_svd_ex = np.sqrt(np.cumsum(s_rand[::-1] ** 2))[::-1]
        r_svd_exact.append(int(np.searchsorted(-res_svd_ex, -eps * norm_exact)))
        _, _, rc_ex = ACA_complet(A_exact_rand, N_test, epsilon=eps)
        r_acac_exact.append(rc_ex)
        u_ex, _ = ACA_partiel(A_exact_rand, N_test, eps)
        r_acap_exact.append(len(u_ex))

        # Cas 2 : Rang numérique (Coulomb)
        res_svd_nu = np.sqrt(np.cumsum(s_c[::-1] ** 2))[::-1]
        r_svd_num.append(int(np.searchsorted(-res_svd_nu, -eps * norm_A_coul)))
        _, _, rc_nu = ACA_complet(A_num_coulomb, N_test, epsilon=eps)
        r_acac_num.append(rc_nu)
        u_nu, _ = ACA_partiel(A_num_coulomb, N_test, eps)
        r_acap_num.append(len(u_nu))

    plt.semilogx(tolerances, r_svd_exact, "k-", lw=2, label="Rang Exact (r=8) - SVD")
    plt.semilogx(tolerances, r_acap_exact, "g:", lw=2, label="Rang Exact (r=8) - ACA Partiel")
    plt.semilogx(tolerances, r_svd_num, "o-", label="Rang Numérique (Coulomb) - SVD")
    plt.semilogx(tolerances, r_acac_num, "s--", label="Rang Numérique (Coulomb) - ACA Complet")
    plt.semilogx(tolerances, r_acap_num, "^:", label="Rang Numérique (Coulomb) - ACA Partiel")
    
    plt.gca().invert_xaxis()
    plt.title("3. Rangs obtenus en fonction de la tolérance $\\epsilon$")
    plt.xlabel("Tolérance $\\epsilon$ (de $10^{-1}$ à $10^{-12}$)")
    plt.ylabel("Rang obtenu $r(\\epsilon)$")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # --------------------------------------------------------------------------
    # FIGURE 4 : Temps de calcul et complexités en fonction de N
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    tailles_N = [100, 200, 400, 700, 1000, 1400, 1800, 2000, 2500, 3000]
    t_svd, t_acac, t_acap = [], [], []
    eps_bench = 1e-6

    for N_val in tailles_N:
        A_bench = matrice_rang_numerique_coulomb(N_val)

        # SVD complète + troncature
        t0 = time.perf_counter()
        U_b, s_b, Vt_b = np.linalg.svd(A_bench, full_matrices=False)
        _ = svd_tronquee(U_b, s_b, Vt_b, r=6)
        t_svd.append(time.perf_counter() - t0)

        # ACA Complet
        t0 = time.perf_counter()
        _, _, _ = ACA_complet(A_bench, rang_final=15, epsilon=eps_bench)
        t_acac.append(time.perf_counter() - t0)

        # ACA Partiel
        t0 = time.perf_counter()
        u_b, v_b = ACA_partiel(A_bench, N_val, eps_bench)
        t_acap.append(time.perf_counter() - t0)

    N_arr = np.array(tailles_N, dtype=float)
    plt.loglog(N_arr, t_svd, "o-", label="SVD Tronquée (mesuré)")
    plt.loglog(N_arr, t_acac, "s-", label="ACA Complet (mesuré)")
    plt.loglog(N_arr, t_acap, "^-", label="ACA Partiel (mesuré)")

    # Pentes théoriques
    plt.loglog(N_arr, t_svd[-1] * (N_arr / N_arr[-1]) ** 3, "k:", alpha=0.6, label="Pente théorique $\\mathcal{O}(N^3)$")
    plt.loglog(N_arr, t_acac[-1] * (N_arr / N_arr[-1]) ** 2, "k--", alpha=0.6, label="Pente théorique $\\mathcal{O}(r N^2)$")
    plt.loglog(N_arr, t_acap[-1] * (N_arr / N_arr[-1]) ** 1, "k-.", alpha=0.6, label="Pente théorique $\\mathcal{O}(r^2 N)$")

    plt.title("4. Temps de calcul et complexités asymptotiques ($\\epsilon = 10^{-6}$)")
    plt.xlabel("Taille de la matrice $N$ (échelle log)")
    plt.ylabel("Temps d'exécution en secondes (échelle log)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()