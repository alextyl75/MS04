import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1


def maillage_segments(N, a, forme):
    # On calcule les angles dans tous les cas
    angles = np.array([2*np.pi*k/N for k in range(0, N)])
    if forme == "cercle":
        points = a * np.array([np.cos(angles), np.sin(angles)])
    elif forme == "carre":
        # 1. On calcule les coordonnées sur le cercle unitaire
        x_cercle = np.cos(angles)
        y_cercle = np.sin(angles)
        
  
        norme_infinie = np.maximum(np.abs(x_cercle), np.abs(y_cercle))
        
        points = a * np.array([x_cercle / norme_infinie, y_cercle / norme_infinie])
        
    elif forme == "etoile":
        # Modulation du rayon avec un cosinus pour créer 5 branches
        # rayon = a * (1 + 0.4 * np.sin(5 * angles))
        n = 5            # Nombre de branches de l'étoile
        R_max = 1.4 * a  # Rayon des pointes (équivalent à 1 + 0.4)
        R_min = 0.6 * a  # Rayon des creux (équivalent à 1 - 0.4)
        
        
        # 1. On "replie" l'angle pour toujours se situer sur un demi-secteur angulaire [0, pi/n]
        pi_sur_n = np.pi / n
        phi = pi_sur_n - np.abs((angles % (2 * pi_sur_n)) - pi_sur_n)
        
        # 2. Application de l'équation polaire de la ligne droite
        numerateur = R_max * R_min * np.sin(pi_sur_n)
        denominateur = R_min * np.sin(pi_sur_n - phi) + R_max * np.sin(phi)
        rayon = numerateur / denominateur
        points = np.array([rayon * np.cos(angles+np.pi/2), rayon * np.sin(angles+np.pi/2)])
    
    points = points.T #mettre au format (N,2)
    segments = [[i, (i + 1) % N] for i in range(N)]
    milieux = 0.5*np.array([[points[i][0]+ points[(i + 1) % N][0], points[i][1]+ points[(i + 1) % N][1] ] for i in range(N)])
    longueurs = np.sqrt(np.array([(points[i][0] - points[(i + 1) % N][0])**2 + (points[i][1] - points[(i + 1) % N][1])**2  for i in range(N)]))
    normales = -np.array([[(points[i][1] - points[(i + 1) % N][1]), ( points[(i + 1) % N][0] - points[i][0])]/longueurs[i]  for i in range(N)])

    return points,segments,milieux,longueurs,normales

def affichage_maillage(points, segments, milieux, longueurs, normales):
    X = points[:,0]
    Y = points[:,1]

    plt.figure(figsize=(12, 12))

    for noeud1, noeud2 in segments:
        plt.plot([X[noeud1], X[noeud2]], [Y[noeud1], Y[noeud2]], color='blue', zorder=1)

    plt.scatter(X, Y, color='red', s=50, zorder=3, label="Noeuds")


    milieux_X = milieux[:, 0]
    milieux_Y = milieux[:, 1]
    plt.scatter(milieux_X, milieux_Y, color='green', s=30, zorder=3, label="Milieux")


    normales_X = normales[:, 0]
    normales_Y = normales[:, 1]
    plt.quiver(milieux_X, milieux_Y, normales_X, normales_Y, 
               color='orange', angles='xy', scale_units='xy', scale=1, 
               width=0.005, zorder=2, label="Normales sortantes")

    ax = plt.gca() 
    ax.set_aspect('equal', adjustable='box')
    

    marge = 0.4
    plt.xlim(np.min(X) - marge, np.max(X) + marge)
    plt.ylim(np.min(Y) - marge, np.max(Y) + marge)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Visualisation du maillage avec normales normalisées")
    plt.show()


def u_diff(r, theta, k, a, N_serie):
    sum = 0
    for n in range(-N_serie, N_serie + 1):
        sum += ((-1j)**n * jv(n, k*a) / hankel1(n, k*a))* hankel1(n, k*r) * np.exp(1j*n*theta)
    return -sum

def trace_u_d(r, theta, k, a, N_serie):
    sum = 0
    for n in range(-N_serie, N_serie + 1):
        sum += ((-1j)**n * jv(n, k*a) / hankel1(n, k*a))* ( hankel1(n-1, k*r) - hankel1(n+1, k*r)) * np.exp(1j*n*theta)
    return -(k/2)*sum

def cylindrique_to_cartesien(r,theta):
    return [r*np.cos(theta),r*np.sin(theta)]

def cartesien_to_cylindrique(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return [r, theta]

def p(r, theta, k, a, N): #N est le nombre de terme de la série
    trace_u_inc = -1j*k*np.cos(theta)*np.exp(-1j*k*r*np.cos(theta))
    return (-trace_u_d(r, theta, k, a, N)-trace_u_inc)




def affiche_p(milieux, k, a, N_serie):
    x = milieux[:, 0]
    y = milieux[:, 1]
    
    r, theta = cartesien_to_cylindrique(x, y)
    
    valeurs_p = p(r, theta, k, a, N_serie)
    
    p_reel = np.real(valeurs_p)
    p_imag = np.imag(valeurs_p)
    
    val_max = max(np.max(np.abs(p_reel)), np.max(np.abs(p_imag)))
    vmin = -val_max
    vmax = val_max
    # ----------------------------------------------------------
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sc1 = ax1.scatter(x, y, c=p_reel, cmap='coolwarm', vmin=vmin, vmax=vmax, s=50, edgecolor='black', zorder=2)
    ax1.plot(x, y, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax1.set_title("Partie Réelle de p")
    ax1.axis('equal')
    ax1.grid(True, linestyle=':', alpha=0.7)
    plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)

    sc2 = ax2.scatter(x, y, c=p_imag, cmap='coolwarm', vmin=vmin, vmax=vmax, s=50, edgecolor='black', zorder=2)
    ax2.plot(x, y, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax2.set_title("Partie Imaginaire de p")
    ax2.axis('equal')
    ax2.grid(True, linestyle=':', alpha=0.7)
    plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
    
    fig.suptitle(f"Visualisation du champ p sur la frontière (k={k}, N={N_serie})", fontsize=14)
    plt.tight_layout()
    plt.show()

# fonctions pour calcul complet bem
def quad_Green(G, x, a, b, nq,k):
    sum = 0
    xi, w = np.polynomial.legendre.leggauss(nq)

    for i in range(nq):
        sum += w[i] * G(x, 0.5 * (a+b) + 0.5 * xi[i] * (b-a),k)

    sum *= 0.5 * np.linalg.norm(a-b)

    return sum




def A(X, points, segments, N, nq,G, k): #quad est une quadrature (tableau de taille n_q) contenant les tableaux poid,point
    n_obs = np.shape(X)[0]
    A = np.zeros((n_obs,N), dtype=complex)
    print("A",np.shape(A))
    print("n_obs",n_obs)
    for i in range(n_obs):
        for j in range(N):
            A[i][j] = quad_Green(G,X[i,:],points[segments[j][0]],points[segments[j][1]],nq,k)
    return A


def calcul_erreur_relative(u_num, u_ref):
    """
    Calcule l'erreur relative en norme infinie : max|u_num - u_ref| / max|u_ref|
    """
    erreur_absolue_max = np.max(np.abs(u_num - u_ref))
    norme_ref_max = np.max(np.abs(u_ref))
    
    return erreur_absolue_max / norme_ref_max

def u(X, points, segments, N, p, nq,G, k):
    return A(X, points, segments, N, nq,G, k) @ p


