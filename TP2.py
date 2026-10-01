# from distutils.log import error
import fonctions
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, hankel1
from scipy.sparse.linalg import gmres
from matplotlib.patches import Polygon

# Constantes du problème
gamma_euler = 0.5772156649
k = 1
nq = 4
nq_tilde = 5
a = 1
N = 100

test_q5 = False
test_erreur_relative = False
test_q6_N = False
test_q6_nq = False
test_q8 = False
test_geom = True

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

def A_assemble_pas_traitement(points, segments, nq, nq_tilde, N, k):
    A = np.zeros((N,N), dtype=complex)
    print("A",np.shape(A))
    print("N", N)

    x_quad, w = np.polynomial.legendre.leggauss(nq)
    x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)

    for i in range(N):

        ai, bi = points[segments[i][0]], points[segments[i][1]]

        for j in range(N):

            Aij = 0
            aj, bj = points[segments[j][0]], points[segments[j][1]]

            for k_index in range(nq):
                for k_tilde_index in range(nq_tilde):
                    Aij += w[k_index] * w_tilde[k_tilde_index] * G(0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai) , 0.5 * (aj+bj) + 0.5 * x_quad_tilde[k_tilde_index] * (bj-aj), k)

            Aij *= 0.5 * np.linalg.norm(bi-ai)
            Aij *= 0.5 * np.linalg.norm(bj-aj)

            
            A[i][j] = Aij

    return A

# Question 5 assemblage de A avec traitement de la singularité 

def A_assemble(points, segments, nq, nq_tilde, N, k):
    A = np.zeros((N,N), dtype=complex)
    print("A",np.shape(A))
    print("N", N)

    x_quad, w = np.polynomial.legendre.leggauss(nq)
    x_quad_tilde, w_tilde = np.polynomial.legendre.leggauss(nq_tilde)

    for i in range(N):

        ai, bi = points[segments[i][0]], points[segments[i][1]]

        for j in range(N):

            Aij = 0

            if i==j: #cas de la singularité gamme_e = gamme_é
                
                # On calcule séparemment la partie singulière et la partie régulière
                reg = 0
                sing = 0

                for k_index in range(nq):
                    for k_tilde_index in range(nq_tilde):
                        reg += w[k_index] * w_tilde[k_tilde_index] * (1j/4 - (1/(2*np.pi) * (np.log(k/2) + gamma_euler)))
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
                    for k_tilde_index in range(nq_tilde):
                        Aij += w[k_index] * w_tilde[k_tilde_index] * G(0.5 * (ai+bi) + 0.5 * x_quad[k_index] * (bi-ai) , 0.5 * (aj+bj) + 0.5 * x_quad_tilde[k_tilde_index] * (bj-aj), k)

                Aij *= 0.5 * np.linalg.norm(bi-ai)
                Aij *= 0.5 * np.linalg.norm(bj-aj)

            
            A[i][j] = Aij

    return A

# comparer intégrale constante vs Legendre

# Comparaison traitements et pas traitements de la singularité
if test_q5:
    points,segments,milieux,longueurs,normales = fonctions.maillage_segments(N, a, "cercle")
    #fonctions.affichage_maillage(points,segments,milieux,longueurs,normales)

    nq_tab = [i for i in range(1,10)]
    nq_tilde_tab = [i+1 for i in nq_tab] 
    error_tab = []

    for i in range(len(nq_tab)):
        A_traitement = A_assemble(points=points, segments=segments, nq=nq_tab[i], nq_tilde=nq_tilde_tab[i], N=N, k=k)
        A_pas_traitement = A_assemble_pas_traitement(points=points, segments=segments, nq=nq_tab[i], nq_tilde=nq_tilde_tab[i], N=N, k=k)

        diag_traitement = np.diag(A_traitement)
        diag_pas_traitement = np.diag(A_pas_traitement)

        print("Diagonale avec traitement :")
        print(diag_traitement)

        print("\nDiagonale sans traitement :")
        print(diag_pas_traitement)

        error_tab.append(np.mean(np.array([np.linalg.norm(diag_traitement[i] -  diag_pas_traitement[i]) for i in range(len(diag_pas_traitement))])))

    print(error_tab)

# Question 6

if test_erreur_relative:

    # On génère A et b

    A = A_assemble(points=points, segments=segments, nq=nq, nq_tilde=nq_tilde, N=N, k=k)
    b = B(points=points, segments=segments, N=N, nq=nq, k=k)

    p_num, info = gmres(A, b, rtol=1e-10)
    p_analytique = fonctions.calcul_p(milieux, k, a, N_serie=20)

    for i in range(N):
        print(
            f"i={i:3d} | "
            f"p_num = {p_num[i]:.6e} | "
            f"p_exact = {p_analytique[i]:.6e} | "
            f"erreur = {abs(p_num[i] - p_analytique[i]):.6e}"
        )

    erreur_relative = (np.linalg.norm(p_num - p_analytique)/ np.linalg.norm(p_analytique))

    print("Erreur relative :", erreur_relative)

    # donnés par chatgpt
    print("info =", info) #info = 0 cv atteinte, info > 0 nb itérations
    print("résidu relatif =", np.linalg.norm(b - A @ p_num) / np.linalg.norm(b)) #residu final

if test_q6_N:
    Nb_test = 10
    a, k = 1, 1
    nq = 5
    nq_tilde = nq

    N_tab = [100*i for i in range(1,Nb_test+1)]
    error_tab = []

    for Npoints in N_tab:
        points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, "cercle")

        A = A_assemble(points=points, segments=segments, nq=nq, nq_tilde=nq_tilde, N=Npoints, k=k)
        b = B(points=points, segments=segments, N=Npoints, nq=nq, k=k)

        p_num, info = gmres(A, b, rtol=1e-10)
        p_analytique = fonctions.calcul_p(milieux, k, a, N_serie=20)

        erreur_relative = (np.linalg.norm(p_num - p_analytique)/ np.linalg.norm(p_analytique))
        error_tab.append(erreur_relative)
        print(N, " ", erreur_relative)
    # graphe test

    plt.figure(figsize=(7, 5))

    plt.loglog(
        N_tab,
        error_tab,
        "o-",
        label="Erreur relative"
    )

    plt.xlabel(r"$N_{\mathrm{points}}$")
    plt.ylabel(r"Erreur relative")

    plt.title("Convergence de la solution numérique\n(Influence de $N$)")

    # Paramètres du test
    plt.text(
        0.97, 0.97,
        rf"$a = {a}$" + "\n" +
        rf"$k = {k}$" + "\n" +
        rf"$n_q = {nq}$",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8
        )
    )

    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.legend()

    plt.tight_layout()
    plt.show()

if test_q6_nq:
    Nb_test = 1
    a, k = 1, 1
    Npoints = 500

    nq_tab = [2*i for i in range(1,Nb_test+1)]
    error_tab = []

    for nq in nq_tab:
        nq_tilde = nq

        points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, "cercle")

        A = A_assemble(points=points, segments=segments, nq=nq, nq_tilde=nq_tilde, N=Npoints, k=k)
        b = B(points=points, segments=segments, N=Npoints, nq=nq, k=k)

        p_num, info = gmres(A, b, rtol=1e-10)
        p_analytique = fonctions.calcul_p(milieux, k, a, N_serie=20)

        erreur_relative = (np.linalg.norm(p_num - p_analytique)/ np.linalg.norm(p_analytique))
        error_tab.append(erreur_relative)
        print(N, " ", erreur_relative)

    plt.figure(figsize=(7, 5))
    plt.loglog(
        nq_tab,
        error_tab,
        "o-"
    )

    plt.xlabel(r"$n_q$")
    plt.ylabel(r"Erreur relative")
    plt.title(r"Erreur relative en fonction de $n_q$")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

        

def p_num(maillage, nq, nq_tilde, Npoints, k):
    points,segments,milieux,longueurs,normales = maillage

    A = A_assemble(points=points, segments=segments, nq=nq, nq_tilde=nq_tilde, N=Npoints, k=k)
    b = B(points=points, segments=segments, N=Npoints, nq=nq, k=k)

    p_num, info = gmres(A, b, rtol=1e-10)

    if info > 0:
        print("Convergence non-atteinte, nb itérations : ", info)

    return(p_num)

# print(p_num(fonctions.maillage_segments(N, a, "cercle"), nq, nq, N, k))

def evaluation_bem_sur_observation_cercle(N, nq, r_obs, a=1, k=5, N_serie=30, N_obs=60):
    """
    Fonction utilitaire qui refait tout le calcul BEM pour un triplet (N, nq, r_obs) donné
    et renvoie l'erreur relative.
    """
    # 1. Maillage obstacle
    maillage = fonctions.maillage_segments(N, a, forme="carre")
    points, segments, milieux, longueurs, normales = maillage
    valeurs_p = p_num(maillage, nq, nq, N, k)
    
    # 2. Maillage observation
    X, _, _, _, _ = fonctions.maillage_segments(N_obs, r_obs, forme="cercle")
    R_obs, Theta_obs = fonctions.cartesien_to_cylindrique(X[:,0], X[:,1])
    
    # 3. Calculs
    u_X = fonctions.u(X, points, segments, N, valeurs_p, nq,G, k) 
    
    u_analytique = fonctions.u_diff(R_obs, Theta_obs, k, a, N_serie)
    
    # 4. Erreur
    return fonctions.calcul_erreur_relative(u_X, u_analytique)


if(test_q8 ==True):

    N=100
    nq = 10
    r_obs = 2
    print(evaluation_bem_sur_observation_cercle(N, nq, r_obs, a=1, k=5, N_serie=30, N_obs=60))





def plot_diffraction_2D_BEM(points, segments, N, p_vals, nq, k, a, L_domaine=5, resolution=80):
    """
    Calcule et affiche le champ total (incident + BEM) sur une grille 2D. avec valeurs de p en arguments
    Attention: Le temps de calcul peut être long car il n'est pas vectorisé.
    """
    print(f"Génération de la grille ({resolution}x{resolution})...")
    
    # 1. Création de la grille spatiale
    x = np.linspace(-L_domaine, L_domaine, resolution)
    y = np.linspace(-L_domaine, L_domaine, resolution)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # # 2. Rayon et Masque Si CERCLE
    # R = np.sqrt(X_grid**2 + Y_grid**2)
    # # On calcule uniquement à l'extérieur stricte (marge de sécurité de 1e-3)
    # mask = R > (a + 1e-3)

    # 2. Rayon (norme infinie pour un carré) et Masque SI CARRE
    # R= np.maximum(np.abs(X_grid), np.abs(Y_grid)) 
    # mask = R > (a + 1e-3)

    # # 2.Masque si ETOILE
    
    # # Passage en coordonnées polaires pour chaque point de la grille
    # R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    # Theta_grid = np.arctan2(Y_grid, X_grid)
    
    # # Calcul du rayon de la frontière de l'étoile pour ces angles
    # R = a * (1 + 0.4 * np.sin(5 * Theta_grid))
    
    # # Le masque est valide pour les points strictement à l'extérieur de l'étoile
    # mask = R_grid > (R + 1e-3)

    ####masque etoile polygone
    # Paramètres de l'étoile
    n = 5            # Nombre de branches de l'étoile
    R_max = 1.4 * a  # Rayon des pointes (équivalent à 1 + 0.4)
    R_min = 0.6 * a  # Rayon des creux (équivalent à 1 - 0.4)
    
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    Theta_grid = np.arctan2(Y_grid, X_grid)-(np.pi/2)
    
    # 1. On "replie" l'angle pour toujours se situer sur un demi-secteur angulaire [0, pi/n]
    pi_sur_n = np.pi / n
    phi = pi_sur_n - np.abs((Theta_grid % (2 * pi_sur_n)) - pi_sur_n)
    
    # 2. Application de l'équation polaire de la ligne droite
    numerateur = R_max * R_min * np.sin(pi_sur_n)
    denominateur = R_min * np.sin(pi_sur_n - phi) + R_max * np.sin(phi)
    R = numerateur / denominateur
    
    # 3. Application du masque
    mask = R_grid > (R + 1e-3)
       
    # # Code utilisé pour générer l'étoile
    # rayon = a * (1 + 0.4 * np.sin(5 * angles))
    # points = np.array([rayon * np.cos(angles), rayon * np.sin(angles)])
    
    # 3. Extraction des coordonnées des points extérieurs sous forme de liste (N_ext, 2)
    X_eval = np.column_stack((X_grid[mask], Y_grid[mask]))
    
    print(f"Calcul BEM sur {len(X_eval)} points extérieurs en cours (cela peut prendre un moment)...")
    
    # 4. Calcul du champ diffusé par la BEM sur ces points
    u_diff_X = fonctions.u(X_eval,points,segments, N, p_vals, nq,G,k)
    
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
    
    # # On dessine l'obstacle
    # cercle = plt.Circle((0, 0), a, color='dimgray', zorder=10)
    # plt.gca().add_patch(cercle)
    # On dessine l'obstacle (carré de côté 2a centré en 0,0)
    # carre = plt.Rectangle((-a, -a), 2*a, 2*a, color='dimgray', zorder=10)
    # plt.gca().add_patch(carre)
    # Génération des 10 angles correspondant aux sommets (5 pointes + 5 creux)
    # On ajoute +1 pour fermer la boucle au point de départ
    angles_poly = np.linspace(0, 2*np.pi, 2*n + 1)
    
    # Alternance des rayons : R_max pour les indices pairs, R_min pour les impairs
    rayons_poly = np.where(np.arange(2*n + 1) % 2 == 0, R_max, R_min)
    
    # Conversion polaire -> cartésien
    x_poly = rayons_poly * np.cos(angles_poly+np.pi/2)
    y_poly = rayons_poly * np.sin(angles_poly+np.pi/2)
    
    # Création et ajout du polygone strict sur le graphe
    sommets = np.column_stack((x_poly, y_poly))
    etoile_patch = Polygon(sommets, color='dimgray', zorder=10)
    plt.gca().add_patch(etoile_patch)

    # # 1. Génération des points du contour de l'étoile (500 points suffisent pour un tracé lisse)
    # angles = np.linspace(0, 2*np.pi, 500)
    
    # # Formule de l'onde exacte pour le rayon
    # rayon = a * (1 + 0.4 * np.sin(5 * angles))#(1 + 0.4 * (2 / np.pi) * np.arcsin(np.sin(5 * angles)))
    
    # # Conversion polaire -> cartésien
    # x_etoile = rayon * np.cos(angles)
    # y_etoile = rayon * np.sin(angles)
    
    # # Regroupement des x et y dans un tableau de coordonnées (N, 2)
    # sommets = np.column_stack((x_etoile, y_etoile))
    
    # # 2. Création et ajout du polygone sur le graphe
    # etoile = Polygon(sommets, color='dimgray', zorder=10)
    # plt.gca().add_patch(etoile)
    # ##fin obstacle etoile

    plt.colorbar(label="Re(u_total) - BEM")
    plt.title(f"Visualisation de la diffraction (BEM)\nk={k}, a={a}, Points BEM: {N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.show()
    print("Affichage terminé !")


if (test_geom == True):
    a=1
    k = 5
    N=100
    nq = 10
    r_obs = 2

    maillage = fonctions.maillage_segments(N, a, forme="etoile")
    points, segments, milieux, longueurs, normales = maillage
    valeurs_p = p_num(maillage, nq, nq, N, k)
    plot_diffraction_2D_BEM(points, segments, N, valeurs_p, nq, k, a, L_domaine=4, resolution=100)

