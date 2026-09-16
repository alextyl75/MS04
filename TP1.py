import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1

# --- Constantes du problème ---
N = 25
a = 1.

def maillage_segments(N, a):
    angles = [2*np.pi*k/N for k in range(0, N)]
    points = a * np.array([np.sin(angles), np.cos(angles)])
    
    # Création directe de la liste des segments avec le modulo N
    segments = [[i, (i + 1) % N] for i in range(N)]
    
    return points, segments

def u_diff(r, theta, k, a, N):
    sum = 0
    for n in range(-N, N + 1):
        sum += ((-1j)**n * jv(n, k*a) / hankel1(n, k*a))* hankel1(n, k*r) * np.exp(1j*n*theta)
    return -sum

# --- Question 1---

# Test fonction implementation somme partielle : on conserve Nserie = 20

N_serie_tab = [1,3,5,10,20,35]
N_serie_tab = [10,20,35,50,100]
k_tab = [1, 5, 10, 20, 50]
erreur_tab = []
erreur = 0
k = 1
Nb_points = 10 
epsilon = 1e-10

N_result = []
j = 1
for k in k_tab:

    erreur_derniere = 0

    for N_serie in N_serie_tab:

        erreur = 0

        for i in range(Nb_points):

            theta = 2 * np.pi * i / Nb_points

            erreur += np.abs(
                np.exp(-1j * k * a * np.cos(theta))
                + u_diff(a, theta, k, a, N_serie)
            )

        erreur_moyenne = erreur / Nb_points
        erreur_derniere = erreur_moyenne

        if erreur_moyenne <= epsilon:

            N_result.append(N_serie)
            erreur_tab.append(erreur_moyenne)

            break

    else:
        N_result.append(N_serie_tab[-1])
        erreur_tab.append(erreur_derniere)

print("rang en fonction de ka :", N_result)
print("derniere erreur enregistrée:", erreur_tab)

# --- Représentation graphique ---
ka_tab = [k * a for k in k_tab]

plt.figure(figsize=(8, 5))

plt.plot(
    ka_tab,
    N_result,
    marker='o',
    linewidth=2
)

plt.xlabel(r"$ka$", fontsize=13)
plt.ylabel(r"Rang $N$", fontsize=13)
plt.title(r"Rang $N$ en fonction de $ka$", fontsize=15)

plt.grid(True, alpha=0.3)

# Paramètres affichés sur le côté
texte = (
    rf"$a = {a}$" "\n"
    rf"$N_{{points}} = {Nb_points}$" "\n"
    rf"$\varepsilon = {epsilon:.0e}$"
)

plt.text(
    1.02,
    0.5,
    texte,
    transform=plt.gca().transAxes,
    verticalalignment='center',
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="black"
    )
)

plt.tight_layout()
plt.show()



# --- Question 2---

points, segments = maillage_segments(N, a)

print("Liste des segments :")
print(segments)

X = points[0]
Y = points[1]

plt.figure(figsize=(6, 6))

# Tracé des arêtes directement à partir de la liste des segments
for noeud1, noeud2 in segments:
    plt.plot([X[noeud1], X[noeud2]], [Y[noeud1], Y[noeud2]], color='blue')

# Tracé des points
plt.scatter(X, Y, color='red', s=50, zorder=2)

plt.axis('equal')
#plt.show()


