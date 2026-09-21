import fonctions

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1

# Question 2

def quad_gauss_legendre(f, a, b, nq):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * f( 0.5 * (a+b) + 0.5 * xi[i] * (b-a) )

    sum *= 0.5 * np.abs(a-b)

    return sum

def integrale_monome(a, b, deg):
    return (b**(deg+1) - a**(deg+1))/(deg+1)

# Point tests 
a = 0
b = 1
deg_test = 10
nq_test = 3

deg_test_tab = [i for i in range(1,deg_test)]
integrale_quad_tab = []
integrale_monome_tab = []


for deg in deg_test_tab:
    def monome(x):
        return x**(deg)

    integrale_quad_tab.append(quad_gauss_legendre(monome, a, b, nq_test))
    integrale_monome_tab.append(integrale_monome(a, b, deg))

error_tab = [np.abs(integrale_quad_tab[i] - integrale_monome_tab[i]) for i in range(len(integrale_quad_tab))]

print(integrale_monome_tab)
print(integrale_quad_tab)
 
#Test

# Test

plt.figure()

plt.semilogy(
    deg_test_tab,
    error_tab,
    'o-',
)

plt.xlabel("Degré du monôme")
plt.ylabel("Erreur absolue")
plt.title(
    f"Erreur de la quadrature en fonction du degré "
    f"($n_q={nq_test}$)"
)

plt.xticks(deg_test_tab)
plt.grid(True, which="both")
plt.legend()

plt.show()


##question 3



points,segments,milieux,longueurs,normales = maillage_segments(N, a, forme="cercle")

def quad_Green(G, x, a, b, nq):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * G(x, 0.5 * (a+b) + 0.5 * xi[i] * (b-a) )

    sum *= 0.5 * np.abs(a-b)

    return sum

def G(x,y):
    return 0

quad, milieux = []
p = np.array()

X = np.array()
def A(X, N,nq): #quad est une quadrature (tableau de taille n_q) contenant les tableaux poid,point
    A = np.array(N,len(X))
    for i in range(len(X)):
        for j in range(N):
            A[i][j] = quad_Green(G,X[i],points[segments[0][J]],points[segments[1][j]],nq) G(x, 0.5* ((points[segments[0][i]] + points[segments[1][i]]) + quad[1][j]*(points[segments[1][i]] - points[segments[0][i]]))) * longueurs[i]/2
def u(X,N,p,nq):
    return A(X,N,nq)*p


    