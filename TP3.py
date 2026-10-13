import fonctions
import numpy as np
import matplotlib.pyplot as plt
import random

test_q12 = False
test_q3 = False
test_q4 = True

def matrice_faible_random(n,m,rang):
    A = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if (i<rang) and (j<rang):
                A[i,j] =  random.random()  # Entre 0.0 et 1.0 (exclu)i**j#
    return A
if test_q12:
    U,sigma,V = np.linalg.svd(matrice_faible_random(10,15,5))
    print(np.shape(sigma),sigma)# une valeur non nulle de plus que le rang analytique ?

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





    