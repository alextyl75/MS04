import fonctions
import numpy as np
import matplotlib.pyplot as plt
import random



test_q1 = False
test_q2 = False
test_q3 = True
test_q4 = True

def matrice_faible_random(n,m,rang):
    A = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if (i<rang) and (j<rang):
                A[i,j] = (i+1)**(j+1) # random.random()  # Entre 0.0 et 1.0 (exclu)
    return A

def svd_tronquee(U, sigma, Vt, r):

    #chat pour cette syntaxe
    Ur = U[:, :r]
    Sigmar = np.diag(sigma[:r])
    Vtr = Vt[:r, :]

    Mr = Ur @ Sigmar @ Vtr

    return Mr

if test_q1:
    A = matrice_faible_random(10,15,5)
    print(A)
    U, sigma, V = np.linalg.svd(A)
    print(np.shape(sigma),sigma)# une valeur non nulle de plus que le rang analytique ?

if test_q2:
    N = 10
    M = 15
    rang_tab = [i for i in range(1, 11)]
    error_tab = []

    A = matrice_faible_random(N, M, N)
    U, sigma, Vt = np.linalg.svd(A)

    for rang in rang_tab :
        M_r = svd_tronquee(U, sigma, Vt, rang)
        error_tab.append(np.linalg.norm(A - M_r))

    plt.semilogy(rang_tab, error_tab, 'o-')
    plt.xlabel("Rang de la SVD tronquée $r$")
    plt.ylabel("Erreur de reconstruction")
    plt.title(r"Erreur de reconstruction $\|M-M_r\|_F$ en fonction du rang de la SVD tronquée")
    plt.grid(True, which="both")

    # Informations affichées à droite du graphique
    m, n = A.shape
    rang_A = np.linalg.matrix_rank(A)

    plt.subplots_adjust(right=0.72)
    plt.text(
        1.05, 0.5,
        f"rang(A) = {rang_A}\nsize(A) = ({m}, {n})",
        transform=plt.gca().transAxes,
        va="center",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.5")
    )

    plt.show()

def ACA_iteration(R, rang_iteration):
    """Fais l'itération de ACA et renvoie u_{k+1} et v_{k+1}^T"""

    m, n = R.shape

    i_star, j_star = np.unravel_index(np.argmax(np.abs(R)), R.shape)
    pivot = R[i_star, j_star]

    if pivot == 0:
        return None, None

    # Nouvelle colonne
    u_k = R[:, j_star] / pivot
    v_k_trans = R[i_star, :]

    return u_k, v_k_trans

def ACA_complet(A, rang_final):
    n, m = A.shape

    A_approx = np.zeros((n, m))
    R = A.copy()

    for rang_iteration in range(1, rang_final + 1):
        u, v = ACA_iteration(R, rang_iteration)

        if u is None:
            break

        A_approx = A_approx + np.outer(u, v)
        R = R - np.outer(u, v)

    return A_approx, R

if test_q3:
    N = 10
    M = 15      

    A = matrice_faible_random(N, M, N)
    U, sigma, Vt = np.linalg.svd(A)

    rang_tab = [i for i in range(6, 11)]
    print(rang_tab)
    error_tab = []

    for rang in rang_tab :
        A_r, R = ACA_complet(A, rang)
        error_tab.append(np.linalg.norm(R))

    print(error_tab)
    print("taille", len(error_tab))
    print("taille", len(rang_tab))

    #plt.semilogy(rang_tab, error_tab, 'o-')
    plt.figure()
    plt.plot(rang_tab, error_tab, 'o-')
    plt.xlabel("Rang $r$")
    plt.ylabel(r"Résidu $\|R_r\|_F$")
    plt.title("Résidu de l'ACA complet en fonction du rang")
    plt.grid(True, which="both")
    plt.show()


def ACA_partiel(A, N, epsilon):
    A_approx = np.zeros((N,N))
    A_approx_norm = 0.
    next_indice = 0
    index_used = []
    cols_used = []
    v_k_used = []
    u_k_used = []
    norm_prod_uv = 0.
    
    def ACA_partiel_step(A_approx, index_ligne):
        res_row = A[index_ligne, :].copy().astype(float)
        for u, v in zip(u_k_used, v_k_used):
            res_row -= u[index_ligne] * v

        abs_res_row = np.abs(res_row)
        abs_res_row[cols_used] = -1 
        j = np.argmax(abs_res_row)
        pivot = res_row[j]

        # On empêche la division par 0.0
        if np.abs(pivot) < 1e-14:
            u_k = np.zeros(N)
            v_k = np.zeros(N)
            update = np.zeros((N,N))
            # On passe à la ligne suivante pour forcer l'algorithme à chercher ailleurs,
            # ou cela forcera l'arrêt naturel via la norme qui vaudra 0
            next_indice_val = (index_ligne + 1) % N 
        else:
            res_col = A[:, j].copy().astype(float)
            for u, v in zip(u_k_used, v_k_used):
                res_col -= u * v[j]
                
            u_k = res_col / pivot
            v_k = res_row
            update = np.outer(u_k, v_k)
            
            cols_used.append(j)
            index_used.append(index_ligne)
            
            col_abs = np.abs(u_k).copy() 
            col_abs[index_used] = -1
            next_indice_val = np.argmax(col_abs)

        v_k_used.append(v_k)
        u_k_used.append(u_k)
        
        return A_approx + update, next_indice_val
    
    step = 1

    while(norm_prod_uv >= epsilon * A_approx_norm):
        A_approx, next_indice = ACA_partiel_step(A_approx, next_indice)

        u_k = u_k_used[-1]
        v_k = v_k_used[-1]
        norm_prod_uv = np.linalg.norm(u_k) * np.linalg.norm(v_k)
        
        sum_norm = 0.
        for l in range(len(u_k_used) - 1):
            sum_norm += np.inner(u_k_used[l], u_k) * np.inner(v_k_used[l], v_k)

        # SÉCURITÉ 2 : On empêche les nombres négatifs sous la racine
        valeur_sous_racine = A_approx_norm**2 + 2*sum_norm + norm_prod_uv**2
        A_approx_norm = np.sqrt(max(0.0, valeur_sous_racine))
        
        step += 1

    print(f"Convergence atteinte en {step - 1} étapes : critère {norm_prod_uv} < {epsilon * A_approx_norm} ")
    return A_approx

if test_q4:
    A = matrice_faible_random(10,10,5)
    print(A)
    U,sigma,V = np.linalg.svd(A)
    A_approx = ACA_partiel(A,N = 10,epsilon=0.01)
    print(A_approx)
    print(np.linalg.norm(A-A_approx)/np.linalg.norm(A))





    