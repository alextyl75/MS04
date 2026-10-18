import time
import matplotlib.pyplot as plt
import numpy as np
import fonctions

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
    A_approx = np.zeros((n, m), dtype=A.dtype)
    R = A.copy()
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
    ACA avec pivotage partiel sur matrice dense.
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
        res_row = A[index_ligne, :].copy()
        for u, v in zip(u_k_used, v_k_used):
            res_row -= u[index_ligne] * v

        abs_res_row = np.abs(res_row)
        abs_res_row[cols_used] = -1
        j = np.argmax(abs_res_row)
        pivot = res_row[j]

        if np.abs(pivot) < 1e-12:
            u_k = np.zeros(N, dtype=A.dtype)
            v_k = np.zeros(N, dtype=A.dtype)
            next_indice_val = (index_ligne + 1) % N
        else:
            res_col = A[:, j].copy()
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
    while step <= N:
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
            sum_norm += np.vdot(u_k_used[l], u_k) * np.vdot(v_k_used[l], v_k)

        valeur_sous_racine = A_approx_norm**2 + 2 * sum_norm.real + norm_prod_uv**2
        A_approx_norm = np.sqrt(max(0.0, valeur_sous_racine))

        # Condition d'arrêt appliquée à partir de la 2e étape
        if step > 1 and (norm_prod_uv < epsilon * A_approx_norm):
            break

        step += 1

    if verbose:
        print(f"Convergence atteinte en {len(u_k_used)} étapes : critère {norm_prod_uv:.2e} < {epsilon * A_approx_norm:.2e}")
    return u_k_used, v_k_used

def reconstruire_ACA_partiel(u_k_used, v_k_used, N=None):
    """Reconstruit la matrice dense à partir des facteurs."""
    if len(u_k_used) == 0:
        if N is None:
            raise ValueError("Préciser N si les listes sont vides.")
        return np.zeros((N, N), dtype=complex)

    n_rows, n_cols = len(u_k_used[0]), len(v_k_used[0])
    A_approx = np.zeros((n_rows, n_cols), dtype=complex)
    for u, v in zip(u_k_used, v_k_used):
        A_approx += np.outer(u, v)
    return A_approx

def calcul_Aij(points, segments, i, j, nq, nq_tilde, k):
    Aij = 0
    x_quad, w = np.polynomial.legendre.leggauss(nq)
    x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)
    ai, bi = points[segments[i][0]], points[segments[i][1]]
    gamma_euler = 0.5772156649

    if i == j:
        reg = 0
        sing = 0
        for k_index in range(nq):
            for k_tilde_index in range(nq_tilde):
                reg += w[k_index] * w_tilde[k_tilde_index] * (1j/4 - (1/(2*np.pi) * (np.log(k/2) + gamma_euler)))
        reg *= (0.5 * np.linalg.norm(bi-ai))**2

        for k_index in range(nq):
            d_b_x = bi - (0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai))
            d_a_x = ai - (0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai))
            Tau_e = (bi - ai)/np.linalg.norm(bi-ai)
            sing += w[k_index] * (np.dot(d_b_x, Tau_e) * np.log(np.linalg.norm(d_b_x)) - np.dot(d_a_x, Tau_e) * np.log(np.linalg.norm(d_a_x)) - np.linalg.norm(bi-ai))

        sing *= -(1/(2*np.pi))
        sing *= (0.5 * np.linalg.norm(bi-ai))
        Aij = sing + reg
    else:
        aj, bj = points[segments[j][0]], points[segments[j][1]]
        for k_index in range(nq):
            for k_tilde_index in range(nq_tilde):
                Aij += w[k_index] * w_tilde[k_tilde_index] * fonctions.G(0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai) , 0.5 * (aj+bj) + 0.5 * x_quad_tilde[k_tilde_index] * (bj-aj), k)
        Aij *= 0.5 * np.linalg.norm(bi-ai)
        Aij *= 0.5 * np.linalg.norm(bj-aj)

    return Aij

def ACA_partiel_free(points, segments, nq, nq_tilde, k, N, epsilon, verbose=False, rmax=None):
    """ACA partiel sans assemblage complet (matrice-free)."""
    A_approx_norm = 0.0
    next_indice = 0
    index_used = []
    cols_used = []
    v_k_used = []
    u_k_used = []
    norm_prod_uv = 0.0

    def ACA_partiel_step(index_ligne):
        res_row = np.array([calcul_Aij(points, segments, index_ligne, j, nq, nq_tilde, k) for j in range(N)], dtype=complex)
        for u, v in zip(u_k_used, v_k_used):
            res_row -= u[index_ligne] * v

        abs_res_row = np.abs(res_row)
        abs_res_row[cols_used] = -1
        j = np.argmax(abs_res_row)
        pivot = res_row[j]

        if np.abs(pivot) < 1e-12:
            u_k = np.zeros(N, dtype=A.dtype)
            v_k = np.zeros(N, dtype=A.dtype)
            next_indice_val = (index_ligne + 1) % N
        else:
            res_col = np.array([calcul_Aij(points, segments, i, j, nq, nq_tilde, k) for i in range(N)], dtype=complex)
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
            sum_norm += np.vdot(u_k_used[l], u_k) * np.vdot(v_k_used[l], v_k)

        valeur_sous_racine = A_approx_norm**2 + 2 * sum_norm.real + norm_prod_uv**2
        A_approx_norm = np.sqrt(max(0.0, valeur_sous_racine))

        # Condition d'arrêt appliquée à partir de la 2e étape
        if step > 1 and (norm_prod_uv < epsilon * A_approx_norm):
            break

        step += 1

    return u_k_used, v_k_used

# ==============================================================================
# 3. SCRIPT PRINCIPAL DE TEST SUR LA MATRICE BEM A
# ==============================================================================

test_a = False
b = False 
c = False
d = True

if __name__ == "__main__":
    plt.rcParams.update({"font.size": 11})
    
    Npoints = 100
    a = 1.0
    k = 1.0
    nq, nq_tilde = 10, 10

    print("--- Génération du maillage et de la matrice BEM dense A ---")
    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, "cercle")
    A = fonctions.A_assemble(points=points, segments=segments, nq=nq, nq_tilde=nq_tilde, N=Npoints, k=k)
    norm_A = np.linalg.norm(A, "fro")

    print("Calcul de la SVD de A...")
    U_A, s_A, Vt_A = np.linalg.svd(A, full_matrices=False)

    # --------------------------------------------------------------------------
    # FIGURE 1 : Spectres des valeurs singulières
    # --------------------------------------------------------------------------

    if test_a :
        plt.figure(figsize=(10, 6))
        k_max = min(50, Npoints)
        k_max = Npoints
        plt.semilogy(range(1, k_max + 1), s_A[:k_max] / s_A[0], "o-", label="Matrice BEM A (Cercle)")
        plt.axhline(1e-6, color="red", linestyle=":", label="Tolérance $\\epsilon = 10^{-6}$")
        plt.title("1. Décroissance des valeurs singulières normalisées de la matrice BEM $A$")
        plt.xlabel("Indice $k$")
        plt.ylabel("$\\sigma_k / \\sigma_1$ (échelle log)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------
    # FIGURE 2 : Erreur relative en fonction du rang
    # --------------------------------------------------------------------------
    if b :
        plt.figure(figsize=(10, 6))
        rangs_test = range(1, min(30, Npoints))
        rangs_test = range(1, Npoints)
        err_svd, err_aca_c = [], []

        for r in rangs_test:
            A_svd = svd_tronquee(U_A, s_A, Vt_A, r)
            err_svd.append(np.linalg.norm(A - A_svd, "fro") / norm_A)

            A_acac, _, _ = ACA_complet(A, rang_final=r)
            err_aca_c.append(np.linalg.norm(A - A_acac, "fro") / norm_A)

        eps_grid = np.logspace(-1, -10, 15)
        rangs_acap_pts, err_acap_pts = [], []
        rangs_acap_pts_free, err_acap_pts_free = [], []
        for eps in eps_grid:
            u_list, v_list = ACA_partiel(A, Npoints, eps)
            A_acap = reconstruire_ACA_partiel(u_list, v_list, N=Npoints)
            rangs_acap_pts.append(len(u_list))
            err_acap_pts.append(np.linalg.norm(A - A_acap, "fro") / norm_A)

            print("je te hais")
            u_list, v_list = ACA_partiel_free(points, segments, nq, nq_tilde, k, Npoints, eps, rmax=None)
            print("je te hais")
            A_acap = reconstruire_ACA_partiel(u_list, v_list, N=Npoints)
            rangs_acap_pts_free.append(len(u_list))
            err_acap_pts_free.append(np.linalg.norm(A - A_acap, "fro") / norm_A)

        print(rangs_acap_pts)
        print(err_acap_pts)
        plt.semilogy(rangs_test, err_svd, "o-", label="SVD Tronquée (optimale)")
        plt.semilogy(rangs_test, err_aca_c, "s--", label="ACA Complet")
        if len(rangs_acap_pts) > 0:
            plt.semilogy(rangs_acap_pts, err_acap_pts, "^:", label="ACA Partiel")
        if len(rangs_acap_pts_free) > 0:
            plt.semilogy(rangs_acap_pts_free, err_acap_pts_free, "d-.", color="purple", label="ACA Partiel Free")

        plt.title("Convergence de l'erreur relative sur la matrice BEM $A$")
        plt.xlabel("Rang de l'approximation $r$")
        plt.ylabel("Erreur relative $\\|A - A_r\\|_F / \\|A\\|_F$")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------
    # FIGURE 3 : Rangs obtenus en fonction de la tolérance
    # --------------------------------------------------------------------------
    if c:
        plt.figure(figsize=(10, 6))
        tolerances = np.logspace(-1, -10, 10)
        r_svd, r_acac, r_acap, r_acap_free = [], [], [], []

        for eps in tolerances:
            res_svd = np.sqrt(np.cumsum(s_A[::-1] ** 2))[::-1]
            r_svd.append(int(np.searchsorted(-res_svd, -eps * norm_A)))
            
            _, _, rc = ACA_complet(A, Npoints, epsilon=eps)
            r_acac.append(rc)
            
            u_p, _ = ACA_partiel(A, Npoints, eps)
            r_acap.append(len(u_p))

            u_p_free, _ = ACA_partiel_free(points, segments, nq, nq_tilde, k, Npoints, eps, rmax=None)
            r_acap_free.append(len(u_p))

        plt.semilogx(tolerances, r_svd, "o-", label="SVD")
        plt.semilogx(tolerances, r_acac, "s--", label="ACA Complet")
        plt.semilogx(tolerances, r_acap, "^:", label="ACA Partiel")
        plt.semilogx(tolerances, r_acap_free, "d-.", label="ACA Partiel free")
        
        plt.gca().invert_xaxis()
        plt.title("Rangs obtenus en fonction de la tolérance $\\epsilon$")
        plt.xlabel("Tolérance $\\epsilon$")
        plt.ylabel("Rang obtenu $r(\\epsilon)$")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------
    # FIGURE 4 : Temps de calcul comparatifs (SVD, ACA complet, ACA partiel, ACA free)
    # --------------------------------------------------------------------------
    if d:
        plt.figure(figsize=(10, 6))
        tailles_N = [200,400,600,800,1000]
        t_svd, t_acac, t_acap, t_free = [], [], [], []
        t_assemblage_seul = []
        eps_bench = 1e-6

        for N_val in tailles_N:
            pts_b, segs_b, _, _, _ = fonctions.maillage_segments(N_val, a, "cercle")
            
            # 1. Temps d'assemblage de la matrice A
            t0 = time.perf_counter()
            A_b = fonctions.A_assemble(points=pts_b, segments=segs_b, nq=nq, nq_tilde=nq_tilde, N=N_val, k=k)
            t_ass = time.perf_counter() - t0
            t_assemblage_seul.append(t_ass)

            # 2. SVD (temps de décomposition seule)
            t0 = time.perf_counter()
            Ub, sb, Vtb = np.linalg.svd(A_b, full_matrices=False)
            _ = svd_tronquee(Ub, sb, Vtb, r=5)
            t_svd.append(t_ass + time.perf_counter() - t0)

            # 3. ACA Complet (temps de compression seul)
            t0 = time.perf_counter()
            _, _, _ = ACA_complet(A_b, rang_final=15, epsilon=eps_bench)
            t_acac.append(t_ass + time.perf_counter() - t0)

            # 4. ACA Partiel (temps de compression seul)
            t0 = time.perf_counter()
            _, _ = ACA_partiel(A_b, N_val, eps_bench)
            t_acap.append(t_ass + time.perf_counter() - t0)

            # 5. ACA Free (calcul des entrées à la volée, sans A pré-assemblée)
            t0 = time.perf_counter()
            _ = ACA_partiel_free(pts_b, segs_b, nq, nq_tilde, k, N_val, eps_bench)
            t_free.append(t_ass + time.perf_counter() - t0)

        N_arr = np.array(tailles_N, dtype=float)

        # Affichage des temps de compression / calcul purs
        plt.loglog(N_arr, t_svd, "o-", label="SVD (décomposition seule)")
        plt.loglog(N_arr, t_acac, "s-", label="ACA Complet (compression seule)")
        plt.loglog(N_arr, t_acap, "^-", label="ACA Partiel (compression seule)")
        plt.loglog(N_arr, t_free, "d-", color="purple", label="ACA Free (à la volée, $O(r^2 N)$ calculs d'éléments)")
        plt.loglog(N_arr, t_assemblage_seul, "k--", alpha=0.5, label="Assemblage complet de $A$ ($O(N^2)$)")

        plt.title("4. Temps de calcul comparatifs purs ($\\epsilon = 10^{-6}$)")
        plt.xlabel("Taille de la matrice $N$ (échelle log)")
        plt.ylabel("Temps d'exécution en secondes (échelle log)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()