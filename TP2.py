#import fonctions 
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1

def uinc(x, k):
    return np.exp(-1j * k * x[0])

def quad_gauss_legendre(f, a, b, nq):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * f( 0.5 * (a+b) + 0.5 * xi[i] * (b-a) )

    sum *= 0.5 * np.linalg.norm(b-a)

    return sum

def quad_Green(G, x, a, b, nq):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * G(x, 0.5 * (a+b) + 0.5 * xi[i] * (b-a))

    sum *= 0.5 * np.linalg.norm(a-b)

    return sum

def G(x,y):
    return 1j*hankel1(0, k*np.linalg.norm(x-y))/4

def A_old(X, points, segments, N, nq, k): #quad est une quadrature (tableau de taille n_q) contenant les tableaux poid,point
    n_obs = np.shape(X)[0]
    A = np.zeros((n_obs,N), dtype=complex)
    print("A",np.shape(A))
    print("n_obs",n_obs)
    for i in range(n_obs):
        for j in range(N):
            A[i][j] = quad_Green(G,X[i,:],points[segments[j][0]],points[segments[j][1]],nq)
    return A

# Question 3

def B(X, points, segments, nq, k):
    n_obs = np.shape(X)[0]
    b = np.zeros(n_obs, dtype=complex)
    print("b", np.shape(b))
    print("n_obs", n_obs)
    for i in range(n_obs):
        b[i] = -quad_gauss_legendre( #chatgpt pour la syntaxe ici
            lambda x: uinc(x, k),
            points[segments[i][0]],
            points[segments[i][1]],
            nq
        )
    return b

#Question 4

def A_assemble(X, points, segments, nq, N, k):
    n_obs = np.shape(X)[0]
    A = np.zeros((n_obs,N), dtype=complex)
    print("A",np.shape(A))
    print("n_obs",n_obs)

    x_quad, w = np.polynomial.legendre.leggauss(nq)

    for i in range(n_obs):

        ai, bi = points[segments[i][0]], points[segments[i][1]]

        for j in range(N):

            Aij = 0
            aj, bj = points[segments[j][0]], points[segments[j][1]]

            for k in range(nq):
                for k_tilde in range(nq):
                    Aij += w[k] * w[k_tilde] * G(0.5 * (ai+bi) + 0.5 * x_quad[k] * (bi-ai) , 0.5 * (aj+bj) + 0.5 * x_quad[k_tilde] * (bj-aj))

            Aij *= 0.5 * np.linalg.norm(bi-ai)
            Aij *= 0.5 * np.linalg.norm(bj-aj)

            A[i][j] = Aij

    return A

