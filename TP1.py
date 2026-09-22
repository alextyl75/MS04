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





def quad_Green(G, x, a, b, nq):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * G(x, 0.5 * (a+b) + 0.5 * xi[i] * (b-a))

    sum *= 0.5 * np.linalg.norm(a-b)

    return sum

def G(x,y):
    return 1j*hankel1(0, k*np.linalg.norm(x-y))/4


def A(X, N, nq): #quad est une quadrature (tableau de taille n_q) contenant les tableaux poid,point
    n_obs = np.shape(X)[0]
    A = np.zeros((n_obs,N), dtype=complex)
    print("A",np.shape(A))
    print("n_obs",n_obs)
    for i in range(n_obs):
        for j in range(N):
            A[i][j] = quad_Green(G,X[i,:],points[segments[j][0]],points[segments[j][1]],nq)
    return A
def u(X,N,p,nq):
    return A(X,N,nq)@p

def calcul_p(milieux, k, a, N_serie):
    x = milieux[:, 0]
    y = milieux[:, 1]
    
    r, theta = fonctions.cartesien_to_cylindrique(x, y)
    
    return fonctions.p(r, theta, k, a, N_serie)


#Question 4 TEST
# a = 1
# N = 100
# N_serie = 30
# k=5
# points,segments,milieux,longueurs,normales = fonctions.maillage_segments(N, a, forme="cercle")
# valeurs_p = calcul_p(milieux,k,a,N_serie)
# print("p",np.shape(valeurs_p))

# print("segmets",np.shape(segments))
# r_obs = 5
# N_obs = 10
# X,_,_,_,_ = fonctions.maillage_segments(N_obs,r_obs, forme="cercle")
# print("X",np.shape(X))
# nq = 4 #ordre de quadrature
# u_X = u(X,N,valeurs_p,nq)
# R_obs,Theta_obs = fonctions.cartesien_to_cylindrique(X[:,0],X[:,1])
# u_analytique = fonctions.u_diff(R_obs,Theta_obs,k,a,N_serie)

# print(np.shape(u_analytique),np.shape(u_X))
# print(u_analytique)
# print(u_X)


# print(u_analytique-u_X)

def calcul_erreur_relative(u_num, u_ref):
    """
    Calcule l'erreur relative en norme infinie : max|u_num - u_ref| / max|u_ref|
    """
    erreur_absolue_max = np.max(np.abs(u_num - u_ref))
    norme_ref_max = np.max(np.abs(u_ref))
    
    return erreur_absolue_max / norme_ref_max

def plot_champ_sur_points(X, u_valeurs, titre="Champ sur les points d'observation"):
    """
    Affiche la partie réelle du champ u évalué sur un nuage de points X en 2D.
    """
    plt.figure(figsize=(7, 6))
    # On colorie les points en fonction de la partie réelle du champ
    sc = plt.scatter(X[:, 0], X[:, 1], c=np.real(u_valeurs), cmap='RdBu_r', s=50, edgecolors='k')
    plt.colorbar(sc, label="Re(u)")
    
    plt.title(titre)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_comparaison_1D_cercle(X, u_num, u_ref):
    """
    Affiche l'amplitude de l'onde selon l'angle. Très pratique si les points X 
    forment un cercle (ce qui est le cas dans ton test).
    """
    angles = np.arctan2(X[:, 1], X[:, 0])
    # On trie les points par angle croissant pour avoir une belle courbe continue
    ordre = np.argsort(angles)
    
    plt.figure(figsize=(8, 5))
    plt.plot(angles[ordre], np.real(u_ref)[ordre], 'k-', linewidth=2, label='Analytique (Re)')
    plt.plot(angles[ordre], np.real(u_num)[ordre], 'r--', marker='o', label='BEM (Re)')
    
    plt.title("Comparaison des parties réelles sur le cercle d'observation")
    plt.xlabel("Angle θ (rad)")
    plt.ylabel("Re(u)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_diffraction_2D(k, a, N_serie, L_domaine=5, resolution=200):
    """
    Calcule et affiche le champ total (incident + diffracté) sur une grille 2D 
    en utilisant la solution analytique.
    """
    # 1. Création de la grille spatiale
    x = np.linspace(-L_domaine, L_domaine, resolution)
    y = np.linspace(-L_domaine, L_domaine, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # 2. Passage en coordonnées cylindriques
    R = np.sqrt(X_grid**2 + Y_grid**2)
    Theta = np.arctan2(Y_grid, X_grid)
    
    # 3. Masquage de l'intérieur du cylindre (le champ y est nul ou non pertinent)
    mask = R >= a
    
    # 4. Calcul du champ diffusé à l'extérieur
    u_diff_champ = np.zeros_like(R, dtype=complex)
    u_diff_champ[mask] = fonctions.u_diff(R[mask], Theta[mask], k, a, N_serie)
    
    # 5. Définition du champ incident 
    # (D'après ton p(), trace_u_inc contient exp(-1j*k*r*cos(theta)), 
    # l'onde plane vient de +x et va vers -x, on fait de même ici)
    u_inc_champ = np.zeros_like(R, dtype=complex)
    u_inc_champ[mask] = np.exp(-1j * k * X_grid[mask])
    
    # 6. Champ Total
    u_tot = u_diff_champ + u_inc_champ
    
    # 7. Visualisation
    plt.figure(figsize=(9, 7))
    # On utilise un "pcolormesh" pour afficher la grille de couleurs
    plt.pcolormesh(X_grid, Y_grid, np.real(u_tot), cmap='RdBu_r', shading='auto', vmin=-2, vmax=2)
    
    # On dessine l'obstacle (le cylindre de rayon a) en gris
    cercle = plt.Circle((0, 0), a, color='dimgray', zorder=10)
    plt.gca().add_patch(cercle)
    
    plt.colorbar(label="Re(u_total)")
    plt.title(f"Visualisation de la diffraction (Analytique)\nk={k}, a={a}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()





# --- TEST ET VISUALISATION ---
a = 1
N = 100       # Nombres de segments sur l'obstacle
N_serie = 30  # Ordre de troncature de la série de Bessel/Hankel
k = 5

# --- 1. Résolution BEM ---
points, segments, milieux, longueurs, normales = fonctions.maillage_segments(N, a, forme="cercle")
valeurs_p = calcul_p(milieux, k, a, N_serie)

# --- 2. Évaluation sur les points d'observation ---
r_obs = 5
N_obs = 60    # On augmente un peu pour faire de belles courbes
X, _, _, _, _ = fonctions.maillage_segments(N_obs, r_obs, forme="cercle")

nq = 4 # ordre de quadrature
u_X = u(X, N, valeurs_p, nq)

# --- 3. Solution Analytique de référence ---
R_obs, Theta_obs = fonctions.cartesien_to_cylindrique(X[:,0], X[:,1])
u_analytique = fonctions.u_diff(R_obs, Theta_obs, k, a, N_serie)

# --- 4. Calcul de l'erreur ---
erreur_rel = calcul_erreur_relative(u_X, u_analytique)
print(f"Erreur relative (Norme infinie) : {erreur_rel:.4e} (soit {erreur_rel*100:.2f}%)")

# --- 5. Visualisations ---
# Plot des points isolés dans l'espace
plot_champ_sur_points(X, u_X, titre="Champ diffusé (BEM) aux points d'observation")

# Comparaison des courbes 1D sur le périmètre d'observation
plot_comparaison_1D_cercle(X, u_X, u_analytique)

# Visualisation 2D de l'onde totale autour de l'objet (peut prendre 1 ou 2 secondes)
plot_diffraction_2D(k, a, N_serie, L_domaine=5, resolution=200)

def plot_diffraction_2D_BEM(points, segments, N, p_vals, nq, k, a, L_domaine=5, resolution=80):
    """
    Calcule et affiche le champ total (incident + BEM) sur une grille 2D.
    Attention: Le temps de calcul peut être long car il n'est pas vectorisé.
    """
    print(f"Génération de la grille ({resolution}x{resolution})...")
    
    # 1. Création de la grille spatiale
    x = np.linspace(-L_domaine, L_domaine, resolution)
    y = np.linspace(-L_domaine, L_domaine, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # 2. Rayon et Masque
    R = np.sqrt(X_grid**2 + Y_grid**2)
    # On calcule uniquement à l'extérieur stricte (marge de sécurité de 1e-3)
    mask = R > (a + 1e-3)
    
    # 3. Extraction des coordonnées des points extérieurs sous forme de liste (N_ext, 2)
    X_eval = np.column_stack((X_grid[mask], Y_grid[mask]))
    
    print(f"Calcul BEM sur {len(X_eval)} points extérieurs en cours (cela peut prendre un moment)...")
    
    # 4. Calcul du champ diffusé par la BEM sur ces points
    u_diff_X = u(X_eval, N, p_vals, nq)
    
    # 5. Ajout du champ incident (même définition que dans la fonction analytique)
    u_inc_X = np.exp(-1j * k * X_eval[:, 0])
    
    # 6. Champ total sur les points valides
    u_tot_X = u_diff_X + u_inc_X
    
    # 7. Reconstitution de la matrice 2D (on met NaN à l'intérieur de l'objet)
    u_tot_champ = np.full(R.shape, np.nan, dtype=complex)
    u_tot_champ[mask] = u_tot_X
    
    # 8. Affichage
    plt.figure(figsize=(9, 7))
    plt.pcolormesh(X_grid, Y_grid, np.real(u_tot_champ), cmap='RdBu_r', shading='auto', vmin=-2, vmax=2)
    
    # On dessine l'obstacle
    cercle = plt.Circle((0, 0), a, color='dimgray', zorder=10)
    plt.gca().add_patch(cercle)
    
    plt.colorbar(label="Re(u_total) - BEM")
    plt.title(f"Visualisation de la diffraction (BEM)\nk={k}, a={a}, Points BEM: {N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()
    print("Affichage terminé !")

# Tracer le champ BEM sur la grille 2D
plot_diffraction_2D_BEM(points, segments, N, valeurs_p, nq, k, a, L_domaine=5, resolution=80)