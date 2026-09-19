import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1

# --- Constantes du problème ---
N = 100
a = 1.

def maillage_segments(N, a, forme):
    # On calcule les angles dans tous les cas
    angles = np.array([2*np.pi*k/N for k in range(0, N)])
    if forme == "cercle":
        points = a * np.array([np.cos(angles), np.sin(angles)])
        
    elif forme == "carre":
        # pts_x = []
        # pts_y = []
        
        # # Idéalement N doit être un multiple de 4 pour une symétrie parfaite.
        # n_cote = N // 4 
        # n_list = [n_cote, n_cote, n_cote, N - 3*n_cote] # Gère le reste si N % 4 != 0
        

        
        # # Côté 2 (Droite) : de (a, -a) vers (a, a) exclus
        # pts_x.extend([a] * n_list[1])
        # pts_y.extend(np.linspace(-a, a, n_list[1], endpoint=False))
        
        # # Côté 3 (Haut) : de (a, a) vers (-a, a) exclus
        # pts_x.extend(np.linspace(a, -a, n_list[2], endpoint=False))
        # pts_y.extend([a] * n_list[2])
        
        # # Côté 4 (Gauche) : de (-a, a) vers (-a, -a) exclus
        # pts_x.extend([-a] * n_list[3])
        # pts_y.extend(np.linspace(a, -a, n_list[3], endpoint=False))

        # # Côté 1 (Bas) : de (-a, -a) vers (a, -a) exclus
        # pts_x.extend(np.linspace(-a, a, n_list[0], endpoint=False))
        # pts_y.extend([-a] * n_list[0])
        
        # points = np.array([pts_x, pts_y])
    


        # 1. On calcule les coordonnées sur le cercle unitaire
        x_cercle = np.cos(angles)
        y_cercle = np.sin(angles)
        
        # 2. On trouve la norme maximale pour chaque point (le plus grand entre |x| et |y|)
        norme_infinie = np.maximum(np.abs(x_cercle), np.abs(y_cercle))
        
        # 3. On divise par cette norme pour ramener les points sur les bords d'un carré
        points = a * np.array([x_cercle / norme_infinie, y_cercle / norme_infinie])
        
    elif forme == "etoile":
        # Modulation du rayon avec un cosinus pour créer 5 branches
        rayon = a * (1 + 0.4 * np.cos(5 * angles))
        points = np.array([rayon * np.cos(angles), rayon * np.sin(angles)])
    
    # Création directe de la liste des segments avec le modulo N
    segments = [[i, (i + 1) % N] for i in range(N)]
    milieux = 0.5*np.array([[points[0][i]+ points[0][(i + 1) % N], points[1][i]+ points[1][(i + 1) % N] ] for i in range(N)])
    longueurs = np.sqrt(np.array([(points[0][i] - points[0][(i + 1) % N])**2 + (points[1][i] - points[1][(i + 1) % N])**2  for i in range(N)]))
    normales = -np.array([[(points[1][i] - points[1][(i + 1) % N]), ( points[0][(i + 1) % N] - points[0][i])]/longueurs[i]  for i in range(N)])

    return points,segments,milieux,longueurs,normales

def affichage_maillage(points, segments, milieux, longueurs, normales):
    X = points[0]
    Y = points[1]

    plt.figure(figsize=(12, 12))

    # 1. Tracé des segments (lignes bleues)
    for noeud1, noeud2 in segments:
        plt.plot([X[noeud1], X[noeud2]], [Y[noeud1], Y[noeud2]], color='blue', zorder=1)

    # 2. Tracé des points / noeuds (rouges)
    plt.scatter(X, Y, color='red', s=50, zorder=3, label="Noeuds")

    # 3. Tracé des milieux (verts)
    milieux_X = milieux[:, 0]
    milieux_Y = milieux[:, 1]
    plt.scatter(milieux_X, milieux_Y, color='green', s=30, zorder=3, label="Milieux")

    # 4. Tracé des normales (flèches oranges)
    normales_X = normales[:, 0]
    normales_Y = normales[:, 1]
    plt.quiver(milieux_X, milieux_Y, normales_X, normales_Y, 
               color='orange', angles='xy', scale_units='xy', scale=1, 
               width=0.005, zorder=2, label="Normales sortantes")

    ax = plt.gca() # Récupère l'axe courant (Get Current Axis)
    ax.set_aspect('equal', adjustable='box')
    
    # On applique la marge fixe
    marge = 0.4
    plt.xlim(np.min(X) - marge, np.max(X) + marge)
    plt.ylim(np.min(Y) - marge, np.max(Y) + marge)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Visualisation du maillage avec normales normalisées")
    plt.show()


def u_diff(r, theta, k, a, N):
    sum = 0
    for n in range(-N, N + 1):
        sum += ((-1j)**n * jv(n, k*a) / hankel1(n, k*a))* hankel1(n, k*r) * np.exp(1j*n*theta)
    return -sum

def trace_u_d(r, theta, k, a, N):
    sum = 0
    for n in range(-N, N + 1):
        sum += ((-1j)**n * jv(n, k*a) / hankel1(n, k*a))* ( hankel1(n-1, k*r) - hankel1(n+1, k*r)) * np.exp(1j*n*theta)
    return -(k/2)*sum

def cylindrique_to_cartesien(r,theta):
    return [r*np.cos(theta),r*np.sin(theta)]

def cartesien_to_cylindrique(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return [r, theta]

def p(r, theta, k, a, N):
    trace_u_inc = -1j*k*np.cos(theta)*np.exp(-1j*k*r*np.cos(theta))
    return (-trace_u_d(r, theta, k, a, N)-trace_u_inc)

import numpy as np
import matplotlib.pyplot as plt

def affiche_p(milieux, k, a, N):
    # 1. Extraction des coordonnées cartésiennes des milieux
    x = milieux[:, 0]
    y = milieux[:, 1]
    
    # 2. Conversion en coordonnées cylindriques
    r, theta = cartesien_to_cylindrique(x, y)
    
    # 3. Évaluation de la fonction p sur tous les milieux
    valeurs_p = p(r, theta, k, a, N)
    
    p_reel = np.real(valeurs_p)
    p_imag = np.imag(valeurs_p)
    
    # --- AJOUT : Calcul des limites globales et symétriques ---
    # On cherche la plus grande valeur absolue pour centrer le 0
    val_max = max(np.max(np.abs(p_reel)), np.max(np.abs(p_imag)))
    vmin = -val_max
    vmax = val_max
    # ----------------------------------------------------------
    
    # 4. Création de la figure à deux volets (1 ligne, 2 colonnes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Sous-graphique 1 : Partie Réelle ---
    # Ajout de vmin et vmax pour fixer l'échelle
    sc1 = ax1.scatter(x, y, c=p_reel, cmap='coolwarm', vmin=vmin, vmax=vmax, s=50, edgecolor='black', zorder=2)
    ax1.plot(x, y, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax1.set_title("Partie Réelle de p")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.7)
    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

    # --- Sous-graphique 2 : Partie Imaginaire ---
    # Ajout de vmin et vmax pour fixer l'échelle (identique au 1er graphique)
    sc2 = ax2.scatter(x, y, c=p_imag, cmap='coolwarm', vmin=vmin, vmax=vmax, s=50, edgecolor='black', zorder=2)
    ax2.plot(x, y, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax2.set_title("Partie Imaginaire de p")
    ax2.axis('equal')
    ax2.grid(True, linestyle=':', alpha=0.7)
    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    
    fig.suptitle(f"Visualisation du champ p sur la frontière (k={k}, N={N})", fontsize=14)
    plt.tight_layout()
    plt.show()

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
