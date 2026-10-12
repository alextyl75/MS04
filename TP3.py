import fonctions
import numpy as np
import matplotlib.pyplot as plt
import random

test_q1 = False
test_q2 = False
test_q3 = True

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

def ACA_ligne(A,index_ligne):
    '''
    Ne fait qu'une itération sur la ligne i
    '''
    j = np.argmax(np.abs(A[index_ligne, :]))
    
    pivot = A[index_ligne, j]
    if pivot == 0:
        print("pivot nul sur cette ligne")
        return A

    update = np.outer(A[:, j] / pivot, A[index_ligne, :])
    return A - update