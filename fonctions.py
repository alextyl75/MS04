import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1


def maillage_segments(N, a, forme):
    # On calcule les angles dans tous les cas
    angles = np.array([2*np.pi*k/N for k in range(0, N)])
    if forme == "cercle":
        points = a * np.array([np.cos(angles), np.sin(angles)])
    elif forme == "carre":
        # 1. On calcule les coordonnées sur le cercle unitaire
        x_cercle = np.cos(angles)
        y_cercle = np.sin(angles)
        
  
        norme_infinie = np.maximum(np.abs(x_cercle), np.abs(y_cercle))
        
        points = a * np.array([x_cercle / norme_infinie, y_cercle / norme_infinie])
        
    elif forme == "etoile":
        # Modulation du rayon avec un cosinus pour créer 5 branches
        # rayon = a * (1 + 0.4 * np.sin(5 * angles))
        n = 5            # Nombre de branches de l'étoile
        R_max = 1.4 * a  # Rayon des pointes (équivalent à 1 + 0.4)
        R_min = 0.6 * a  # Rayon des creux (équivalent à 1 - 0.4)
        
        
        # 1. On "replie" l'angle pour toujours se situer sur un demi-secteur angulaire [0, pi/n]
        pi_sur_n = np.pi / n
        phi = pi_sur_n - np.abs((angles % (2 * pi_sur_n)) - pi_sur_n)
        
        # 2. Application de l'équation polaire de la ligne droite
        numerateur = R_max * R_min * np.sin(pi_sur_n)
        denominateur = R_min * np.sin(pi_sur_n - phi) + R_max * np.sin(phi)
        rayon = numerateur / denominateur
        points = np.array([rayon * np.cos(angles+np.pi/2), rayon * np.sin(angles+np.pi/2)])
    
    points = points.T #mettre au format (N,2)
    segments = [[i, (i + 1) % N] for i in range(N)]
    milieux = 0.5*np.array([[points[i][0]+ points[(i + 1) % N][0], points[i][1]+ points[(i + 1) % N][1] ] for i in range(N)])
    longueurs = np.sqrt(np.array([(points[i][0] - points[(i + 1) % N][0])**2 + (points[i][1] - points[(i + 1) % N][1])**2  for i in range(N)]))
    normales = -np.array([[(points[i][1] - points[(i + 1) % N][1]), ( points[(i + 1) % N][0] - points[i][0])]/longueurs[i]  for i in range(N)])

    return points,segments,milieux,longueurs,normales

def affichage_maillage(points, segments, milieux, longueurs, normales):
    X = points[:,0]
    Y = points[:,1]

    plt.figure(figsize=(12, 12))

    for noeud1, noeud2 in segments:
        plt.plot([X[noeud1], X[noeud2]], [Y[noeud1], Y[noeud2]], color='blue', zorder=1)

    plt.scatter(X, Y, color='red', s=50, zorder=3, label="Noeuds")


    milieux_X = milieux[:, 0]
    milieux_Y = milieux[:, 1]
    plt.scatter(milieux_X, milieux_Y, color='green', s=30, zorder=3, label="Milieux")


    normales_X = normales[:, 0]
    normales_Y = normales[:, 1]
    plt.quiver(milieux_X, milieux_Y, normales_X, normales_Y, 
               color='orange', angles='xy', scale_units='xy', scale=1, 
               width=0.005, zorder=2, label="Normales sortantes")

    ax = plt.gca() 
    ax.set_aspect('equal', adjustable='box')
    

    marge = 0.4
    plt.xlim(np.min(X) - marge, np.max(X) + marge)
    plt.ylim(np.min(Y) - marge, np.max(Y) + marge)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Visualisation du maillage avec normales normalisées")
    plt.show()


def u_diff(r, theta, k, a, N_serie):
    sum = 0
    for n in range(-N_serie, N_serie + 1):
        sum += ((-1j)**n * jv(n, k*a) / hankel1(n, k*a))* hankel1(n, k*r) * np.exp(1j*n*theta)
    return -sum

def trace_u_d(r, theta, k, a, N_serie):
    sum = 0
    for n in range(-N_serie, N_serie + 1):
        sum += ((-1j)**n * jv(n, k*a) / hankel1(n, k*a))* ( hankel1(n-1, k*r) - hankel1(n+1, k*r)) * np.exp(1j*n*theta)
    return -(k/2)*sum

def cylindrique_to_cartesien(r,theta):
    return [r*np.cos(theta),r*np.sin(theta)]

def cartesien_to_cylindrique(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return [r, theta]

def p(r, theta, k, a, N): #N est le nombre de terme de la série
    trace_u_inc = -1j*k*np.cos(theta)*np.exp(-1j*k*r*np.cos(theta))
    return (-trace_u_d(r, theta, k, a, N)-trace_u_inc)

def calcul_p(milieux, k, a, N_serie):
    x = milieux[:, 0]
    y = milieux[:, 1]
    
    r, theta = cartesien_to_cylindrique(x, y)
    
    return p(r, theta, k, a, N_serie)

def affiche_p(milieux, k, a, N_serie):
    x = milieux[:, 0]
    y = milieux[:, 1]
    
    r, theta = cartesien_to_cylindrique(x, y)
    
    valeurs_p = p(r, theta, k, a, N_serie)
    
    p_reel = np.real(valeurs_p)
    p_imag = np.imag(valeurs_p)
    
    val_max = max(np.max(np.abs(p_reel)), np.max(np.abs(p_imag)))
    vmin = -val_max
    vmax = val_max
    # ----------------------------------------------------------
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sc1 = ax1.scatter(x, y, c=p_reel, cmap='coolwarm', vmin=vmin, vmax=vmax, s=50, edgecolor='black', zorder=2)
    ax1.plot(x, y, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax1.set_title("Partie Réelle de p")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.7)
    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

    sc2 = ax2.scatter(x, y, c=p_imag, cmap='coolwarm', vmin=vmin, vmax=vmax, s=50, edgecolor='black', zorder=2)
    ax2.plot(x, y, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax2.set_title("Partie Imaginaire de p")
    ax2.axis('equal')
    ax2.grid(True, linestyle=':', alpha=0.7)
    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    
    fig.suptitle(f"Visualisation du champ p sur la frontière (k={k}, N={N_serie})", fontsize=14)
    plt.tight_layout()
    plt.show()

# fonctions pour calcul complet bem
def quad_Green(G, x, a, b, nq,k):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * G(x, 0.5 * (a+b) + 0.5 * xi[i] * (b-a),k)

    sum *= 0.5 * np.linalg.norm(a-b)

    return sum

def A(X, points, segments, N, nq,G, k): #quad est une quadrature (tableau de taille n_q) contenant les tableaux poid,point
    n_obs = np.shape(X)[0]
    A = np.zeros((n_obs,N), dtype=complex)
    print("A",np.shape(A))
    print("n_obs",n_obs)
    for i in range(n_obs):
        for j in range(N):
            A[i][j] = quad_Green(G,X[i,:],points[segments[j][0]],points[segments[j][1]],nq,k)
    return A


def calcul_erreur_relative(u_num, u_ref):
    """
    Calcule l'erreur relative en norme infinie : max|u_num - u_ref| / max|u_ref|
    """
    erreur_absolue_max = np.max(np.abs(u_num - u_ref))
    norme_ref_max = np.max(np.abs(u_ref))
    
    return erreur_absolue_max / norme_ref_max

def u(X, points, segments, N, p, nq,G, k):
    return A(X, points, segments, N, nq,G, k) @ p

def uinc(x, k):
    return np.exp(-1j * k * x[0])

def quad_gauss_legendre(f, a, b, nq):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * f( 0.5 * (a+b) + 0.5 * xi[i] * (b-a) )

    sum *= 0.5 * np.linalg.norm(b-a)

    return sum

def quad_Green(G, x, a, b, nq, k):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * G(x, 0.5 * (a+b) + 0.5 * xi[i] * (b-a), k)

    sum *= 0.5 * np.linalg.norm(a-b)

    return sum

def G(x,y,k):
    return 1j*hankel1(0, k*np.linalg.norm(x-y))/4

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

def A_assemble(points, segments, nq, nq_tilde, N, k):
    # A = np.zeros((N,N), dtype=complex)
    # print("A",np.shape(A))
    # print("N", N)

    # gamma_euler = 0.5772156649

    # x_quad, w = np.polynomial.legendre.leggauss(nq)
    # x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)

    # print("debut")
    # for i in range(N):

    #     ai, bi = points[segments[i][0]], points[segments[i][1]]

    #     for j in range(N):

    #         Aij = 0

    #         if i==j: #cas de la singularité gamme_e = gamme_é
                
    #             # On calcule séparemment la partie singulière et la partie régulière
    #             reg = 0
    #             sing = 0

    #             for k_index in range(nq):
    #                 for k_tilde_index in range(nq_tilde):
    #                     reg += w[k_index] * w_tilde[k_tilde_index] * (1j/4 - (1/(2*np.pi) * (np.log(k/2) + gamma_euler)))
    #             reg *= (0.5 * np.linalg.norm(bi-ai))**2

    #             # On intégre ensuite la singularité analytiquement l'intégrale au sens de y puis on intégre au sens de x grâce à une quad
    #             for k_index in range(nq):
    #                 d_b_x = bi - (0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai))
    #                 d_a_x = ai - (0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai))
    #                 Tau_e = (bi - ai)/np.linalg.norm(bi-ai)

    #                 sing += w[k_index] * (np.dot(d_b_x, Tau_e) * np.log(np.linalg.norm(d_b_x)) - np.dot(d_a_x, Tau_e) * np.log(np.linalg.norm(d_a_x)) - np.linalg.norm(bi-ai))

    #             sing *= -(1/(2*np.pi))
    #             sing *= (0.5 * np.linalg.norm(bi-ai))

    #             Aij = sing + reg

    #         else :
    #             aj, bj = points[segments[j][0]], points[segments[j][1]]

    #             for k_index in range(nq):
    #                 for k_tilde_index in range(nq_tilde):
    #                     Aij += w[k_index] * w_tilde[k_tilde_index] * G(0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai) , 0.5 * (aj+bj) + 0.5 * x_quad_tilde[k_tilde_index] * (bj-aj), k)

    #             Aij *= 0.5 * np.linalg.norm(bi-ai)
    #             Aij *= 0.5 * np.linalg.norm(bj-aj)

            
    #         A[i][j] = Aij
    # print("fin")

    A = np.zeros((N, N), dtype=complex)
    print("A", np.shape(A))
    print("N", N)

    gamma_euler = 0.5772156649

    x_quad, w = np.polynomial.legendre.leggauss(nq)
    x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)

    # --- OPTIMISATION 1 : Pré-calcul des poids combinés ---
    # Pour la partie régulière (produit tensoriel des poids de quadrature)
    W_reg_matrix = np.outer(w, w_tilde) # Matrice de taille nq x nq_tilde
    somme_w_reg = np.sum(W_reg_matrix)

    print("debut")
    for i in range(N):
        ai, bi = points[segments[i][0]], points[segments[i][1]]
        length_i = np.linalg.norm(bi - ai)
        
        # Points de quadrature sur le segment i (pré-calculés pour la boucle i)
        # x_pts shape: (nq, 2)
        x_pts = 0.5 * (ai + bi) + 0.5 * x_quad[:, np.newaxis] * (bi - ai)

        for j in range(N):
            aj, bj = points[segments[j][0]], points[segments[j][1]]
            length_j = np.linalg.norm(bj - aj)

            if i == j:
                # --- CAS DIAGONAL (Singularité) ---
                # Partie régulière (corrigée de la boucle inutile)
                const_reg = 1j/4 - (1 / (2 * np.pi) * (np.log(k/2) + gamma_euler))
                reg = somme_w_reg * const_reg * (0.5 * length_i)**2

                # Partie singulière intégrée analytiquement en y
                Tau_e = (bi - ai) / length_i
                sing = 0
                for k_index in range(nq):
                    d_b_x = bi - x_pts[k_index]
                    d_a_x = ai - x_pts[k_index]
                    
                    term = (np.dot(d_b_x, Tau_e) * np.log(np.linalg.norm(d_b_x)) - 
                            np.dot(d_a_x, Tau_e) * np.log(np.linalg.norm(d_a_x)) - 
                            length_i)
                    sing += w[k_index] * term

                sing *= -(1 / (2 * np.pi)) * (0.5 * length_i)

                A[i, j] = sing + reg

            else:
                # --- CAS HORS-DIAGONALE ---
                # Points de quadrature sur le segment j
                y_pts = 0.5 * (aj + bj) + 0.5 * x_quad_tilde[:, np.newaxis] * (bj - aj) # shape: (nq_tilde, 2)
                
                # Double somme de quadrature optimisée via numpy broadcasting si possible, 
                # ou boucles épurées
                Aij = 0
                for k_index in range(nq):
                    for k_tilde_index in range(nq_tilde):
                        val_G = G(x_pts[k_index], y_pts[k_tilde_index], k)
                        Aij += W_reg_matrix[k_index, k_tilde_index] * val_G

                Aij *= 0.5 * length_i * 0.5 * length_j
                A[i, j] = Aij

    print("fin")

    return A

def ACA_complet_tolerance(A, tol, k_max=None):
    """
    Factorisation ACA avec pivotage complet et tolérance adaptative.
    """
    n, m = A.shape
    if k_max is None:
        k_max = min(n, m)

    R = A.copy().astype(complex)
    u_list = []
    v_list = []

    norm_A_approx_sq = 0.0  # Frobenius

    for k in range(1, k_max + 1):
        # Recherche du pivot maximal (Pivotage complet)
        i_star, j_star = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        pivot = R[i_star, j_star]

        # Si le résidu restant est nul, on s'arrête
        if np.abs(pivot) < 1e-15:
            break

        # Extraire la colonne et la ligne de résidus
        u_k = R[:, j_star] / pivot
        v_k_trans = R[i_star, :]

        # Mise à jour du résidu
        outer_uv = np.outer(u_k, v_k_trans)
        R = R - outer_uv

        u_list.append(u_k)
        v_list.append(v_k_trans)

        # Calcul des normes pour le critère d'arrêt
        norm_u = np.linalg.norm(u_k)
        norm_v = np.linalg.norm(v_k_trans)
        norm_uv = norm_u * norm_v  # Norme de la nouvelle contribution

        # Mise à jour de l'estimation de la norme de Frobenius de A_approx
        norm_A_approx_sq += norm_uv**2
        for l in range(k - 1):
            prod_u = np.vdot(u_list[l], u_k)
            prod_v = np.vdot(v_list[l], v_k_trans)
            norm_A_approx_sq += 2 * np.real(prod_u * prod_v)

        norm_A_approx = np.sqrt(max(0.0, norm_A_approx_sq))

        # --- CRITÈRE D'ARRÊT ADAPTATIF ---
        if norm_uv <= tol * norm_A_approx:
            break

    rang_obtenu = len(u_list)

    return u_list, v_list, rang_obtenu

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



