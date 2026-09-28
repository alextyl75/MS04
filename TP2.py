import fonctions
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1

# Constantes du problème
gamma_euler = 0.5772156649
k = 1
nq = 10
a = 1
N = 100

# Fonctions du maillage

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

# Question 3

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

#Question 4 sans traitement de la singularité

def A_assemble_pas_traitement(points, segments, nq, N, k):
    A = np.zeros((N,N), dtype=complex)
    print("A",np.shape(A))
    print("N", N)

    x_quad, w = np.polynomial.legendre.leggauss(nq)

    for i in range(N):

        ai, bi = points[segments[i][0]], points[segments[i][1]]

        for j in range(N):

            Aij = 0
            aj, bj = points[segments[j][0]], points[segments[j][1]]

            for k_index in range(nq):
                for k_tilde_index in range(nq):
                    Aij += w[k_index] * w[k_tilde_index] * G(0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai) , 0.5 * (aj+bj) + 0.5 * x_quad[k_tilde_index] * (bj-aj), k)

            Aij *= 0.5 * np.linalg.norm(bi-ai)
            Aij *= 0.5 * np.linalg.norm(bj-aj)

            
            A[i][j] = Aij

    return A

# Question 5 assemblage de A avec traitement de la singularité 

def A_assemble(points, segments, nq, N, k):
    A = np.zeros((N,N), dtype=complex)
    print("A",np.shape(A))
    print("N", N)

    x_quad, w = np.polynomial.legendre.leggauss(nq)

    for i in range(N):

        ai, bi = points[segments[i][0]], points[segments[i][1]]

        for j in range(N):

            Aij = 0

            if i==j: #cas de la singularité gamme_e = gamme_é
                
                # On calcule séparemment la partie singulière et la partie régulière
                reg = 0
                sing = 0

                for k_index in range(nq):
                    for k_tilde_index in range(nq):
                        reg += w[k_index] * w[k_tilde_index] * (1j/4 - (1/(2*np.pi) * (np.log(k/2) + gamma_euler)))
                reg *= (0.5 * np.linalg.norm(bi-ai))**2

                # On intégre ensuite la singularité analytiquement l'intégrale au sens de y puis on intégre au sens de x grâce à une quad
                for k_index in range(nq):
                    d_b_x = bi - (0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai))
                    d_a_x = ai - (0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai))
                    Tau_e = (bi - ai)/np.linalg.norm(bi-ai)

                    sing += w[k_index] * (np.dot(d_b_x, Tau_e) * np.log(np.linalg.norm(d_b_x)) - np.dot(d_a_x, Tau_e) * np.log(np.linalg.norm(d_a_x)) - np.linalg.norm(bi-ai))

                sing *= -(1/(2*np.pi))
                sing *= (0.5 * np.linalg.norm(bi-ai))

                Aij = sing + reg

            else :
                aj, bj = points[segments[j][0]], points[segments[j][1]]

                for k_index in range(nq):
                    for k_tilde_index in range(nq):
                        Aij += w[k_index] * w[k_tilde_index] * G(0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai) , 0.5 * (aj+bj) + 0.5 * x_quad[k_tilde_index] * (bj-aj), k)

                Aij *= 0.5 * np.linalg.norm(bi-ai)
                Aij *= 0.5 * np.linalg.norm(bj-aj)

            
            A[i][j] = Aij

    return A

# Tests de la question 5

points,segments,milieux,longueurs,normales = fonctions.maillage_segments(N, a, "cercle")
#fonctions.affichage_maillage(points,segments,milieux,longueurs,normales)

# comparer intégrale constante vs Legendre

# Comparaison traitements et pas traitements de la singularité
A_traitement = A_assemble(points=points, segments=segments, nq=nq, N=N, k=k)
A_pas_traitement = A_assemble_pas_traitement(points=points, segments=segments, nq=nq, N=N, k=k)

diag_traitement = np.diag(A_traitement)
diag_pas_traitement = np.diag(A_pas_traitement)

print("Diagonale avec traitement :")
print(diag_traitement)

print("\nDiagonale sans traitement :")
print(diag_pas_traitement)

