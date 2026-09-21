import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1
import fonctions

# --- Constantes du problème ---
N = 100
a = 1.


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
#plt.show()

# --- Question 2---

points,segments,milieux,longueurs,normales = maillage_segments(N, a, forme="cercle")

print("Liste des segments :")
print(segments)
affichage_maillage(points,segments,milieux,longueurs,0.2*normales)



# --- Question 3---

h_tab = [10**(-i) for i in range(1, 9)]

r_tab = [1 + i/10 for i in range(10)]

N_serie = 15

erreur_tab = []

theta_tab = [2*np.pi*i/10 for i in range(10)]

k = 5

a = 1

for h in h_tab:

    erreur = 0

    for r in r_tab:

        for theta in theta_tab:

            approx_derive = (
                u_diff(r + h, theta, k, a, N_serie)
                - u_diff(r, theta, k, a, N_serie)
            ) / (h)

            erreur += np.abs(
                approx_derive
                - trace_u_d(r, theta, k, a, N_serie)
            )

    erreur /= len(r_tab) * len(theta_tab)

    erreur_tab.append(erreur)

print(h_tab)
print(erreur_tab)

# --- Représentation de l'erreur ---
plt.figure()

plt.loglog(h_tab, erreur_tab, 'o-')

plt.xlabel(r"$h$")
plt.ylabel(r"Erreur")

plt.title(
    r"Erreur $\left|p-\frac{u^+(r+h,\theta)-u^+(r,\theta)}{h}\right|$ en fonction de h"
)

# Informations sur les paramètres
plt.text(
    0.97, 0.97,
    rf"$a = {a}$" + "\n" +
    rf"$k = {k}$" + "\n" +
    rf"$N_{{serie}} = {N_serie}$",
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)

plt.grid(True, which="both")

plt.show()

##Q4
affiche_p(milieux, k, a, N)
