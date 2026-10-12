import fonctions
import numpy as np
import matplotlib.pyplot as plt
import random

test_q12 = False
test_q3 = True

def matrice_faible_random(n,m,rang):
    A = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            if (i<=rang) and (j<=rang):
                A[i,j] = i**j# random.random()  # Entre 0.0 et 1.0 (exclu)
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