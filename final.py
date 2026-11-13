import math
from typing import List, Optional
import numpy as np
from scipy.special import hankel1
import os
from concurrent.futures import ThreadPoolExecutor

GAMMA_EULER = 0.57721566490153286


# ==========================================
# 1. ARBRE DES CLUSTERS
# ==========================================

class Noeud:
    def __init__(
        self,
        box: list,
        list_pts: list,
        niveau: int,
        indice_local: int,
        fils1: "Noeud" = None,
        fils2: "Noeud" = None,
        diam: float = 0.0,
    ):
        self.box = box
        self.list_pts = list_pts
        self.niveau = niveau
        self.indice_local = indice_local
        self.fils1 = fils1
        self.fils2 = fils2
        self.diam = diam

    @property
    def is_leaf(self):
        """Retourne True si le nœud est une feuille."""
        return self.fils1 is None and self.fils2 is None

    @property
    def n_points(self):
        """Nombre de degrés de liberté dans le cluster."""
        return len(self.list_pts)


def construction_arbre(Npoints, Nleaf, points, strat):
    """Renvoie l'arbre des clusters sous forme d'une liste de nœuds."""

    def calculer_box_et_diam(pts_indexed):
        x_coords = [pt[1][0] for pt in pts_indexed]
        y_coords = [pt[1][1] for pt in pts_indexed]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        box = [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]]
        diam = math.hypot(x_max - x_min, y_max - y_min)
        return box, diam

    def iteration(noeud):
        cluster.append(noeud)

        if len(noeud.list_pts) >= Nleaf:
            xmin, ymin = noeud.box[0]
            xmax, ymax = noeud.box[3]

            Lx = xmax - xmin
            Ly = ymax - ymin
            list_pts1, list_pts2 = [], []

            if strat == "geo":
                if Lx >= Ly:
                    xmid = 0.5 * (xmax + xmin)
                    for pts in noeud.list_pts:
                        if pts[1][0] <= xmid:
                            list_pts1.append(pts)
                        else:
                            list_pts2.append(pts)
                else:
                    ymid = 0.5 * (ymax + ymin)
                    for pts in noeud.list_pts:
                        if pts[1][1] <= ymid:
                            list_pts1.append(pts)
                        else:
                            list_pts2.append(pts)

            elif strat == "mediane":
                alpha = 0 if Lx >= Ly else 1
                pts_tries = sorted(noeud.list_pts, key=lambda pt: pt[1][alpha])
                mid = len(pts_tries) // 2
                list_pts1 = pts_tries[:mid]
                list_pts2 = pts_tries[mid:]

            box_1, diam_1 = calculer_box_et_diam(list_pts1)
            box_2, diam_2 = calculer_box_et_diam(list_pts2)

            fils1 = Noeud(box_1, list_pts1, niveau=noeud.niveau + 1, indice_local=0, diam=diam_1)
            fils2 = Noeud(box_2, list_pts2, niveau=noeud.niveau + 1, indice_local=1, diam=diam_2)

            noeud.fils1 = fils1
            noeud.fils2 = fils2

            iteration(fils1)
            iteration(fils2)

    cluster = []
    points_indexed = [(i, pt) for i, pt in enumerate(points)]
    box, diam = calculer_box_et_diam(points_indexed)
    racine = Noeud(box, points_indexed, niveau=0, indice_local=0, diam=diam)
    iteration(racine)

    return cluster


# ==========================================
# 2. ARBRE DES BLOCS & ADMISSIBILITÉ
# ==========================================

def distance_box(Noeud1, Noeud2):
    x1_min, y1_min = Noeud1.box[0]
    x1_max, y1_max = Noeud1.box[3]

    x2_min, y2_min = Noeud2.box[0]
    x2_max, y2_max = Noeud2.box[3]

    delta_x = x1_min - x2_max if x1_min > x2_max else (x2_min - x1_max if x2_min > x1_max else 0.0)
    delta_y = y1_min - y2_max if y1_min > y2_max else (y2_min - y1_max if y2_min > y1_max else 0.0)

    return math.hypot(delta_x, delta_y)


def is_admissible(noeudX, noeudY, eta):
    diam_min = noeudX.diam if noeudX.diam < noeudY.diam else noeudY.diam
    return diam_min <= eta * distance_box(noeudX, noeudY)


class Noeud_bloc:
    def __init__(
        self,
        tau: Noeud,
        sigma: Noeud,
        niveau: int,
        indice_local: int,
        fils_list: Optional[List["Noeud_bloc"]] = None,
        type_feuille: str = None,
    ):
        self.tau = tau
        self.sigma = sigma
        self.niveau = niveau
        self.indice_local = indice_local
        self.fils_list = fils_list
        self.type_feuille = type_feuille

    @property
    def is_leaf(self):
        """Retourne True si le nœud est une feuille."""
        return self.fils_list is None or len(self.fils_list) == 0

    def diviser(self):
        """Crée les sous-blocs d'interaction enfants valides."""
        enfants_tau = [f for f in (self.tau.fils1, self.tau.fils2) if f is not None]
        enfants_sigma = [f for f in (self.sigma.fils1, self.sigma.fils2) if f is not None]

        self.fils_list = [
            Noeud_bloc(
                tau=child_tau,
                sigma=child_sigma,
                niveau=self.niveau + 1,
                indice_local=i,
            )
            for i, (child_tau, child_sigma) in enumerate(
                (t, s) for t in enfants_tau for s in enfants_sigma
            )
        ]


def arbre_bloc_construction(arbre_cluster, eta):
    arbre_bloc = []

    def iteration(noeud_b):
        arbre_bloc.append(noeud_b)

        if is_admissible(noeud_b.tau, noeud_b.sigma, eta):
            noeud_b.type_feuille = "feuille_admissible"
        elif noeud_b.tau.is_leaf or noeud_b.sigma.is_leaf:
            noeud_b.type_feuille = "feuille_dense"
        else:
            noeud_b.diviser()
            for fils in noeud_b.fils_list:
                iteration(fils)

    racine_cluster = arbre_cluster[0] if isinstance(arbre_cluster, (list, tuple)) else arbre_cluster
    iteration(Noeud_bloc(racine_cluster, racine_cluster, 0, 0))
    return arbre_bloc


# ==========================================
# 3. NOYAU, QUADRATURE, ACA & H-MATRICE
# ==========================================

def G(x, y, k):
    return 0.25j * hankel1(0, k * np.linalg.norm(x - y))


def _terme_singulier(ai, bi, vec, norm_seg, demi_norm, x_quad, w, k):
    """Calcule la contribution diagonale singulière d'un segment."""
    pts = 0.5 * (ai + bi) + 0.5 * x_quad[:, None] * vec[None, :]
    Tau_e = vec / norm_seg

    d_b_x = bi[None, :] - pts
    d_a_x = ai[None, :] - pts

    norm_b = np.hypot(d_b_x[:, 0], d_b_x[:, 1])
    norm_a = np.hypot(d_a_x[:, 0], d_a_x[:, 1])

    termes = (
        (d_b_x @ Tau_e) * np.log(norm_b)
        - (d_a_x @ Tau_e) * np.log(norm_a)
        - norm_seg
    )
    sing = -(1.0 / (2.0 * np.pi)) * demi_norm * np.dot(w, termes)
    constante_reg = 4.0 * (0.25j - (1.0 / (2.0 * np.pi)) * (np.log(0.5 * k) + GAMMA_EULER)) * (demi_norm ** 2)
    return sing + constante_reg


def calcul_ligne(points, segments, indices, i, x_quad, w, x_quad_tilde, w_tilde, k, fonctions=None):
    """Calcule la i-ème ligne locale du bloc défini par 'indices' (vectorisé)."""
    idx_tau = np.asarray(indices[0]).ravel()
    idx_sigma = np.asarray(indices[1]).ravel()
    M = len(idx_sigma)

    ig = idx_tau[i]
    ai = points[segments[ig, 0]]
    bi = points[segments[ig, 1]]
    vec_i = bi - ai
    norm_i = np.hypot(vec_i[0], vec_i[1])
    demi_norm_i = 0.5 * norm_i

    pts_i = 0.5 * (ai + bi) + 0.5 * x_quad[:, None] * vec_i[None, :]

    ligne = np.empty(M, dtype=np.complex128)
    mask_diag = (idx_sigma == ig)
    mask_reg = ~mask_diag

    if np.any(mask_reg):
        jg_reg = idx_sigma[mask_reg]
        aj = points[segments[jg_reg, 0]]
        bj = points[segments[jg_reg, 1]]
        vec_j = bj - aj
        demi_norm_j = 0.5 * np.hypot(vec_j[:, 0], vec_j[:, 1])

        pts_j = 0.5 * (aj + bj)[:, None, :] + 0.5 * x_quad_tilde[None, :, None] * vec_j[:, None, :]

        diff = pts_i[None, :, None, :] - pts_j[:, None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])

        G_vals = 0.25j * hankel1(0, k * dist)
        w_2d = np.outer(w, w_tilde).ravel()
        Aij = G_vals.reshape(len(jg_reg), -1) @ w_2d

        ligne[mask_reg] = Aij * (demi_norm_i * demi_norm_j)

    if np.any(mask_diag):
        ligne[mask_diag] = _terme_singulier(ai, bi, vec_i, norm_i, demi_norm_i, x_quad, w, k)

    return ligne


def calcul_colonne(points, segments, indices, j, x_quad, w, x_quad_tilde, w_tilde, k, fonctions=None):
    """Calcule la j-ème colonne locale du bloc défini par 'indices' (vectorisé)."""
    idx_tau = np.asarray(indices[0]).ravel()
    idx_sigma = np.asarray(indices[1]).ravel()
    N = len(idx_tau)

    jg = idx_sigma[j]
    aj = points[segments[jg, 0]]
    bj = points[segments[jg, 1]]
    vec_j = bj - aj
    norm_j = np.hypot(vec_j[0], vec_j[1])
    demi_norm_j = 0.5 * norm_j

    pts_j = 0.5 * (aj + bj) + 0.5 * x_quad_tilde[:, None] * vec_j[None, :]

    colonne = np.empty(N, dtype=np.complex128)
    mask_diag = (idx_tau == jg)
    mask_reg = ~mask_diag

    if np.any(mask_reg):
        ig_reg = idx_tau[mask_reg]
        ai = points[segments[ig_reg, 0]]
        bi = points[segments[ig_reg, 1]]
        vec_i = bi - ai
        demi_norm_i = 0.5 * np.hypot(vec_i[:, 0], vec_i[:, 1])

        pts_i = 0.5 * (ai + bi)[:, None, :] + 0.5 * x_quad[None, :, None] * vec_i[:, None, :]

        diff = pts_i[:, :, None, :] - pts_j[None, None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])

        G_vals = 0.25j * hankel1(0, k * dist)
        w_2d = np.outer(w, w_tilde).ravel()
        Aij = G_vals.reshape(len(ig_reg), -1) @ w_2d

        colonne[mask_reg] = Aij * (demi_norm_i * demi_norm_j)

    if np.any(mask_diag):
        colonne[mask_diag] = _terme_singulier(aj, bj, vec_j, norm_j, demi_norm_j, x_quad, w, k)

    return colonne


def A_assemble_bloc(points, segments, indices, x_quad, w, x_quad_tilde, w_tilde, k, fonctions=None):
    """Assemblage 100 % vectorisé d'un bloc dense défini par indices = np.ix_(idx_tau, idx_sigma)."""
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)
    idx_tau = np.asarray(indices[0], dtype=np.int64).ravel()
    idx_sigma = np.asarray(indices[1], dtype=np.int64).ravel()

    N, M = len(idx_tau), len(idx_sigma)
    bloc = np.empty((N, M), dtype=np.complex128)

    a_tau = points[segments[idx_tau, 0]]
    b_tau = points[segments[idx_tau, 1]]
    vec_tau = b_tau - a_tau
    norm_tau = np.hypot(vec_tau[:, 0], vec_tau[:, 1])
    demi_norm_tau = 0.5 * norm_tau
    pts_tau = 0.5 * (a_tau + b_tau)[:, None, :] + 0.5 * x_quad[None, :, None] * vec_tau[:, None, :]

    a_sigma = points[segments[idx_sigma, 0]]
    b_sigma = points[segments[idx_sigma, 1]]
    vec_sigma = b_sigma - a_sigma
    norm_sigma = np.hypot(vec_sigma[:, 0], vec_sigma[:, 1])
    demi_norm_sigma = 0.5 * norm_sigma
    pts_sigma = 0.5 * (a_sigma + b_sigma)[:, None, :] + 0.5 * x_quad_tilde[None, :, None] * vec_sigma[:, None, :]

    mask_diag = (idx_tau[:, None] == idx_sigma[None, :])
    mask_reg = ~mask_diag

    if np.any(mask_reg):
        rows_reg, cols_reg = np.nonzero(mask_reg)
        diff = pts_tau[rows_reg][:, :, None, :] - pts_sigma[cols_reg][:, None, :, :]
        dist = np.hypot(diff[..., 0], diff[..., 1])

        G_vals = 0.25j * hankel1(0, k * dist)
        w_2d = np.outer(w, w_tilde).ravel()
        Aij = G_vals.reshape(len(rows_reg), -1) @ w_2d
        bloc[rows_reg, cols_reg] = Aij * (demi_norm_tau[rows_reg] * demi_norm_sigma[cols_reg])

    if np.any(mask_diag):
        rows_diag, cols_diag = np.nonzero(mask_diag)
        for r_loc, c_loc in zip(rows_diag, cols_diag):
            bloc[r_loc, c_loc] = _terme_singulier(
                a_tau[r_loc], b_tau[r_loc], vec_tau[r_loc], norm_tau[r_loc], demi_norm_tau[r_loc], x_quad, w, k
            )

    return bloc


def ACA_partiel_free_bloc(points, segments, k, indices, epsilon, x_quad, w, x_quad_tilde, w_tilde, fonctions=None):
    """ACA partiel sans assemblage complet (matrix-free), optimisé avec pré-calcul géométrique du bloc.
    Renvoie (u_k_used, v_k_used, r) tels que A_bloc ≈ U @ V.T
    """
    idx_tau = np.asarray(indices[0]).ravel()
    idx_sigma = np.asarray(indices[1]).ravel()
    N, M = len(idx_tau), len(idx_sigma)
    max_rank = min(N, M)

    w_2d = np.outer(w, w_tilde).ravel()

    a_tau = points[segments[idx_tau, 0]]
    b_tau = points[segments[idx_tau, 1]]
    vec_tau = b_tau - a_tau
    norm_tau = np.hypot(vec_tau[:, 0], vec_tau[:, 1])
    demi_norm_tau = 0.5 * norm_tau
    pts_tau = 0.5 * (a_tau + b_tau)[:, None, :] + 0.5 * x_quad[None, :, None] * vec_tau[:, None, :]

    a_sigma = points[segments[idx_sigma, 0]]
    b_sigma = points[segments[idx_sigma, 1]]
    vec_sigma = b_sigma - a_sigma
    norm_sigma = np.hypot(vec_sigma[:, 0], vec_sigma[:, 1])
    demi_norm_sigma = 0.5 * norm_sigma
    pts_sigma = 0.5 * (a_sigma + b_sigma)[:, None, :] + 0.5 * x_quad_tilde[None, :, None] * vec_sigma[:, None, :]

    def _fast_row(i_loc):
        ig = idx_tau[i_loc]
        row = np.empty(M, dtype=np.complex128)
        mask_diag = (idx_sigma == ig)
        mask_reg = ~mask_diag

        if np.any(mask_reg):
            diff = pts_tau[i_loc][None, :, None, :] - pts_sigma[mask_reg][:, None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            G_vals = 0.25j * hankel1(0, k * dist)
            row[mask_reg] = (G_vals.reshape(np.count_nonzero(mask_reg), -1) @ w_2d) * (
                demi_norm_tau[i_loc] * demi_norm_sigma[mask_reg]
            )
        if np.any(mask_diag):
            row[mask_diag] = _terme_singulier(
                a_tau[i_loc], b_tau[i_loc], vec_tau[i_loc], norm_tau[i_loc], demi_norm_tau[i_loc], x_quad, w, k
            )
        return row

    def _fast_col(j_loc):
        jg = idx_sigma[j_loc]
        col = np.empty(N, dtype=np.complex128)
        mask_diag = (idx_tau == jg)
        mask_reg = ~mask_diag

        if np.any(mask_reg):
            diff = pts_tau[mask_reg][:, :, None, :] - pts_sigma[j_loc][None, None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            G_vals = 0.25j * hankel1(0, k * dist)
            col[mask_reg] = (G_vals.reshape(np.count_nonzero(mask_reg), -1) @ w_2d) * (
                demi_norm_tau[mask_reg] * demi_norm_sigma[j_loc]
            )
        if np.any(mask_diag):
            col[mask_diag] = _terme_singulier(
                a_sigma[j_loc], b_sigma[j_loc], vec_sigma[j_loc], norm_sigma[j_loc], demi_norm_sigma[j_loc], x_quad, w, k
            )
        return col

    A_approx_norm_sq = 0.0
    next_indice = 0
    rows_used = np.zeros(N, dtype=bool)
    cols_used = np.zeros(M, dtype=bool)
    v_k_used = []
    u_k_used = []

    for _ in range(1, max_rank + 1):
        i_cur = next_indice
        res_row = _fast_row(i_cur)

        if u_k_used:
            U_mat = np.column_stack(u_k_used)
            V_mat = np.vstack(v_k_used)
            res_row -= U_mat[i_cur] @ V_mat

        abs_res_row = np.abs(res_row)
        abs_res_row[cols_used] = -1.0
        j = int(np.argmax(abs_res_row))
        pivot = res_row[j]

        rows_used[i_cur] = True

        if np.abs(pivot) < 1e-12:
            rem = np.flatnonzero(~rows_used)
            if len(rem) == 0:
                break
            next_indice = int(rem[0])
            continue

        res_col = _fast_col(j)
        if u_k_used:
            res_col -= U_mat @ V_mat[:, j]

        u_k = res_col / pivot
        v_k = res_row
        cols_used[j] = True

        norm_prod_uv = np.linalg.norm(u_k) * np.linalg.norm(v_k)
        if norm_prod_uv == 0.0:
            break

        if u_k_used:
            dots_u = np.conj(U_mat.T) @ u_k
            dots_v = np.conj(V_mat) @ v_k
            sum_norm = np.dot(dots_u, dots_v)
            A_approx_norm_sq += 2.0 * sum_norm.real + norm_prod_uv ** 2
        else:
            A_approx_norm_sq = norm_prod_uv ** 2

        u_k_used.append(u_k)
        v_k_used.append(v_k)

        A_approx_norm = np.sqrt(max(0.0, A_approx_norm_sq))
        if norm_prod_uv < epsilon * A_approx_norm:
            break

        rem_rows = np.flatnonzero(~rows_used)
        if len(rem_rows) == 0:
            break
        col_abs = np.abs(u_k)
        col_abs[rows_used] = -1.0
        next_indice = int(np.argmax(col_abs))

    return u_k_used, v_k_used, len(u_k_used)


def H_matrix_light_old(
    arbre_bloc,
    aca_function,
    tol,
    points,
    segments,
    k,
    x_quad,
    w,
    x_quad_tilde,
    w_tilde,
    fonctions=None,
):
    """Construit la H-matrice en mono-cœur avec pré-calcul géométrique global et vectorisation."""
    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)
    x_quad = np.asarray(x_quad, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    x_quad_tilde = np.asarray(x_quad_tilde, dtype=np.float64)
    w_tilde = np.asarray(w_tilde, dtype=np.float64)

    # Pré-calcul géométrique global (une seule fois pour tous les segments)
    a_all = points[segments[:, 0]]
    b_all = points[segments[:, 1]]
    vec_all = b_all - a_all
    norm_all = np.hypot(vec_all[:, 0], vec_all[:, 1])
    demi_norm_all = 0.5 * norm_all
    mid_all = 0.5 * (a_all + b_all)

    pts_i_all = mid_all[:, None, :] + 0.5 * x_quad[None, :, None] * vec_all[:, None, :]
    pts_j_all = mid_all[:, None, :] + 0.5 * x_quad_tilde[None, :, None] * vec_all[:, None, :]
    w_2d = np.outer(w, w_tilde).ravel()

    H = {}

    def parcourt(noeud_b):
        if noeud_b.is_leaf:
            tau = noeud_b.tau
            sigma = noeud_b.sigma

            idx_tau = np.array([p[0] for p in tau.list_pts], dtype=np.int64)
            idx_sigma = np.array([p[0] for p in sigma.list_pts], dtype=np.int64)

            cle = (tuple(idx_tau.tolist()), tuple(idx_sigma.tolist()))

            if noeud_b.type_feuille == "feuille_admissible":
                indices = np.ix_(idx_tau, idx_sigma)
                u_list, v_list, r = aca_function(
                    points, segments, k, indices, tol, x_quad, w, x_quad_tilde, w_tilde, fonctions
                )
                if r > 0:
                    U = np.column_stack(u_list)
                    V = np.column_stack(v_list)
                else:
                    U = np.zeros((len(idx_tau), 0), dtype=np.complex128)
                    V = np.zeros((len(idx_sigma), 0), dtype=np.complex128)
                H[cle] = (U, V)

            elif noeud_b.type_feuille == "feuille_dense":
                N, M = len(idx_tau), len(idx_sigma)
                bloc = np.empty((N, M), dtype=np.complex128)

                mask_diag = (idx_tau[:, None] == idx_sigma[None, :])
                mask_reg = ~mask_diag

                if np.any(mask_reg):
                    rows_reg, cols_reg = np.nonzero(mask_reg)
                    ig_reg = idx_tau[rows_reg]
                    jg_reg = idx_sigma[cols_reg]

                    diff = pts_i_all[ig_reg][:, :, None, :] - pts_j_all[jg_reg][:, None, :, :]
                    dist = np.hypot(diff[..., 0], diff[..., 1])

                    G_vals = 0.25j * hankel1(0, k * dist)
                    Aij = G_vals.reshape(len(rows_reg), -1) @ w_2d
                    bloc[rows_reg, cols_reg] = Aij * (demi_norm_all[ig_reg] * demi_norm_all[jg_reg])

                if np.any(mask_diag):
                    rows_diag, cols_diag = np.nonzero(mask_diag)
                    for r_loc, c_loc in zip(rows_diag, cols_diag):
                        ig = idx_tau[r_loc]
                        bloc[r_loc, c_loc] = _terme_singulier(
                            points[segments[ig, 0]],
                            points[segments[ig, 1]],
                            vec_all[ig],
                            norm_all[ig],
                            demi_norm_all[ig],
                            x_quad,
                            w,
                            k,
                        )

                H[cle] = bloc
        else:
            for fils in noeud_b.fils_list:
                parcourt(fils)

    racine = arbre_bloc[0] if isinstance(arbre_bloc, list) else arbre_bloc
    parcourt(racine)

    return H


def Hmatvec(H, x):
    """Produit matrice-vecteur rapide y = H @ x."""
    y = np.zeros(len(x), dtype=np.complex128)
    for (idx_tau, idx_sigma), bloc in H.items():
        rows = np.asarray(idx_tau, dtype=np.int64)
        cols = np.asarray(idx_sigma, dtype=np.int64)
        if isinstance(bloc, tuple):
            U, V = bloc
            y[rows] += U @ (V.T @ x[cols])
        else:
            y[rows] += bloc @ x[cols]
    return y



# Contexte global initialisé une seule fois par processus worker
_H_CTX = {}


def _init_hmatrix_worker(
    points, segments, pts_i_all, pts_j_all, demi_norm_all, norm_all, vec_all,
    x_quad, w, x_quad_tilde, w_tilde, w_2d, k, tol, aca_function, fonctions
):
    _H_CTX.clear()
    _H_CTX.update({
        "points": points,
        "segments": segments,
        "pts_i_all": pts_i_all,
        "pts_j_all": pts_j_all,
        "demi_norm_all": demi_norm_all,
        "norm_all": norm_all,
        "vec_all": vec_all,
        "x_quad": x_quad,
        "w": w,
        "x_quad_tilde": x_quad_tilde,
        "w_tilde": w_tilde,
        "w_2d": w_2d,
        "k": k,
        "tol": tol,
        "aca_function": aca_function,
        "fonctions": fonctions,
    })


def _traiter_un_bloc(tache):
    """Calcule un bloc feuille (admissible ou dense) en utilisant le contexte partagé."""
    type_feuille, idx_tau, idx_sigma, cle = tache
    ctx = _H_CTX

    if type_feuille == "feuille_admissible":
        indices = np.ix_(idx_tau, idx_sigma)
        u_list, v_list, r = ctx["aca_function"](
            ctx["points"],
            ctx["segments"],
            ctx["k"],
            indices,
            ctx["tol"],
            ctx["x_quad"],
            ctx["w"],
            ctx["x_quad_tilde"],
            ctx["w_tilde"],
            ctx["fonctions"],
        )
        if r > 0:
            U = np.column_stack(u_list)
            V = np.column_stack(v_list)
        else:
            U = np.zeros((len(idx_tau), 0), dtype=np.complex128)
            V = np.zeros((len(idx_sigma), 0), dtype=np.complex128)
        return cle, (U, V)

    else:  # "feuille_dense"
        N, M = len(idx_tau), len(idx_sigma)
        bloc = np.empty((N, M), dtype=np.complex128)

        mask_diag = (idx_tau[:, None] == idx_sigma[None, :])
        mask_reg = ~mask_diag

        if np.any(mask_reg):
            rows_reg, cols_reg = np.nonzero(mask_reg)
            ig_reg = idx_tau[rows_reg]
            jg_reg = idx_sigma[cols_reg]

            diff = ctx["pts_i_all"][ig_reg][:, :, None, :] - ctx["pts_j_all"][jg_reg][:, None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])

            G_vals = 0.25j * hankel1(0, ctx["k"] * dist)
            Aij = G_vals.reshape(len(rows_reg), -1) @ ctx["w_2d"]
            bloc[rows_reg, cols_reg] = Aij * (ctx["demi_norm_all"][ig_reg] * ctx["demi_norm_all"][jg_reg])

        if np.any(mask_diag):
            rows_diag, cols_diag = np.nonzero(mask_diag)
            points = ctx["points"]
            segments = ctx["segments"]
            for r_loc, c_loc in zip(rows_diag, cols_diag):
                ig = idx_tau[r_loc]
                bloc[r_loc, c_loc] = _terme_singulier(
                    points[segments[ig, 0]],
                    points[segments[ig, 1]],
                    ctx["vec_all"][ig],
                    ctx["norm_all"][ig],
                    ctx["demi_norm_all"][ig],
                    ctx["x_quad"],
                    ctx["w"],
                    ctx["k"],
                )

        return cle, bloc

def H_matrix_light(
    arbre_bloc,
    aca_function,
    tol,
    points,
    segments,
    k,
    x_quad,
    w,
    x_quad_tilde,
    w_tilde,
    fonctions=None,
    n_workers=None,
    seuil_points=1500,
):
    """Construit la H-matrice en parallèle avec des threads."""

    points = np.asarray(points, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.int64)

    x_quad = np.asarray(x_quad, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    x_quad_tilde = np.asarray(x_quad_tilde, dtype=np.float64)
    w_tilde = np.asarray(w_tilde, dtype=np.float64)

    # 1. Pré-calcul géométrique global

    a_all = points[segments[:, 0]]
    b_all = points[segments[:, 1]]

    vec_all = b_all - a_all
    norm_all = np.hypot(vec_all[:, 0], vec_all[:, 1])
    demi_norm_all = 0.5 * norm_all

    mid_all = 0.5 * (a_all + b_all)

    pts_i_all = (
        mid_all[:, None, :]
        + 0.5 * x_quad[None, :, None] * vec_all[:, None, :]
    )

    pts_j_all = (
        mid_all[:, None, :]
        + 0.5 * x_quad_tilde[None, :, None] * vec_all[:, None, :]
    )

    w_2d = np.outer(w, w_tilde).ravel()

    # 2. Collecte de toutes les feuilles de l'arbre des blocs

    taches = []

    def collecter(noeud_b):

        if noeud_b.is_leaf:

            idx_tau = np.array(
                [p[0] for p in noeud_b.tau.list_pts],
                dtype=np.int64
            )

            idx_sigma = np.array(
                [p[0] for p in noeud_b.sigma.list_pts],
                dtype=np.int64
            )

            cle = (
                tuple(idx_tau.tolist()),
                tuple(idx_sigma.tolist())
            )

            taches.append(
                (noeud_b.type_feuille, idx_tau, idx_sigma, cle)
            )

        else:

            for fils in noeud_b.fils_list:
                collecter(fils)

    racine = arbre_bloc[0] if isinstance(arbre_bloc, list) else arbre_bloc

    collecter(racine)

    # Tri par aire décroissante pour équilibrer la charge

    taches.sort(
        key=lambda t: len(t[1]) * len(t[2]),
        reverse=True
    )

    # 3. Initialisation du contexte partagé

    init_args = (
        points,
        segments,
        pts_i_all,
        pts_j_all,
        demi_norm_all,
        norm_all,
        vec_all,
        x_quad,
        w,
        x_quad_tilde,
        w_tilde,
        w_2d,
        k,
        tol,
        aca_function,
        fonctions
    )

    _init_hmatrix_worker(*init_args)

    # 4. Choix du nombre de threads

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    n_workers = max(1, int(n_workers))

    # 5. Version séquentielle si nécessaire

    if len(points) < seuil_points or n_workers == 1:

        return dict(
            _traiter_un_bloc(tache)
            for tache in taches
        )

    # 6. Version parallèle avec threads

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        resultats = executor.map(
            _traiter_un_bloc,
            taches
        )

        H = dict(resultats)

    return H

# second membre BEM

def uinc(x, k):
    return np.exp(-1j * k * x[0])

def quad_gauss_legendre(f, a, b, nq):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * f( 0.5 * (a+b) + 0.5 * xi[i] * (b-a) )

    sum *= 0.5 * np.linalg.norm(b-a)

    return sum

def B(points, segments, N, nq, k):
    b = np.zeros(N, dtype=complex)
    print("b", np.shape(b))
    print("N", N)
    for i in range(N):
        b[i] = -quad_gauss_legendre( #chatgpt pour la syntaxe ici
            lambda x: uinc(x, k),
            points[segments[i][0]],
            points[segments[i][1]],
            nq
        )
    return b
