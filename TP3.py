import fonctions
import numpy as np
import matplotlib.pyplot as plt
import random

test_q1 = False
test_q2 = True
test_q3 = False

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

def ACA_complet(A):
    '''
    Ne fait qu'une itération
    '''
    i, j = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    
    pivot = A[i, j]
    update = np.outer(A[:, j] / pivot, A[i, :])
    return A - update

if test_q3:
    A = matrice_faible_random(5,5,3)
    print(A)
    A = ACA_complet(A)
    print(A)

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