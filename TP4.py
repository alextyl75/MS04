import fonctions
import arbre_fonctions
import numpy as np
import matplotlib.pyplot as plt

test_affichage_terminal = False
test_affichage_plt = False
test_q3 = False
test_q4 = False
test_q5_hauteur = False
test_q5_feuille = False

# Parametres
Npoints = 150
a = 1
Nleaf = 5

# Generation du maillage
points,segments,milieux,longueurs,normales = fonctions.maillage_segments(Npoints, a, forme="cercle")
    
# Construction de l'arbre de clusters

# Question 1 et 2
liste_clusters = arbre_fonctions.construction_arbre(Npoints, Nleaf, points)
noeud_racine = liste_clusters[0]

# Affichage textuel dans le terminal
if test_affichage_terminal:
    print(f"PARAMETRES : a={a}, Npoints={Npoints} et Nleaf={Nleaf}.")
    print("--- STRUCTURE DE L'ARBRE ---")
    arbre_fonctions.afficher_arbre_terminal(noeud_racine)

# Affichage des clusters
if test_affichage_plt:
    arbre_fonctions.affichage_clusters(points=points,
        segments=segments,
        liste_clusters=liste_clusters,
        Npoints=Npoints,
        Nleaf=Nleaf,
        a=a,
        afficher_ids=True
    )

# Question 3 (test de validité)
if test_q3:
    print("Test de validité : clusters cercle vs maillage cercle ")
    arbre_fonctions.test_validite(liste_clusters, points)

    #affichage matplotlib
    arbre_fonctions.affichage_clusters(points=points,
            segments=segments,
            liste_clusters=liste_clusters,
            Npoints=Npoints,
            Nleaf=Nleaf,
            a=a,
            afficher_ids=True
    )

    # On teste les clusters pour les cercles avec le maillage en étoile
    print("Test de validité : clusters cercle vs maillage etoile ")
    points_etoile, *_ = fonctions.maillage_segments(Npoints, a, forme="etoile")
    arbre_fonctions.affichage_clusters(points=points_etoile,
                segments=segments,
                liste_clusters=liste_clusters,
                Npoints=Npoints,
                Nleaf=Nleaf,
                a=a,
                afficher_ids=True
    )
    arbre_fonctions.test_validite(liste_clusters, points_etoile)

if test_q4:
    print("permutation", arbre_fonctions.permutation_indice(liste_clusters))
    arbre_fonctions.verification_indices_consecutifs(liste_clusters, points)

if test_q5_hauteur:
    a = 1
    Npoints = 200
    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="cercle")

    Nleaf_tab = [100, 75, 50, 25, 10, 5]
    profondeur_tab = []

    for Nleaf in Nleaf_tab:
        liste_clusters = arbre_fonctions.construction_arbre(Npoints, Nleaf, points)
        profondeur_tab.append(arbre_fonctions.hauteur(liste_clusters))

    # Calcul de log2(Npoints) - log2(Nleaf)
    hauteur_theorique = [np.log2(Npoints) - np.log2(Nleaf) for Nleaf in Nleaf_tab]

    # Tracé du graphique
    plt.figure(figsize=(8, 5))
    plt.plot(Nleaf_tab, profondeur_tab, 'o-', label="Profondeur mesurée")
    plt.plot(
        Nleaf_tab, 
        hauteur_theorique, 
        's--', 
        label=rf"$\log_2(N_{{\mathrm{{points}}}}) - \log_2(N_{{\mathrm{{leaf}}}})$"
    )

    plt.xlabel(r"$N_{\mathrm{leaf}}$")
    plt.ylabel("Hauteur de l'arbre")
    plt.title(rf"Hauteur de l'arbre en fonction de $N_{{\mathrm{{leaf}}}}$ ($a = {a}$, $N_{{\mathrm{{points}}}} = {Npoints}$)")
    plt.grid(True)
    plt.legend()
    plt.show()

if test_q5_feuille:
    a = 1
    Npoints = 200
    points, segments, milieux, longueurs, normales = fonctions.maillage_segments(Npoints, a, forme="cercle")

    Nleaf_tab = [100, 75, 50, 25, 10, 5]
    feuille_tab = []

    for Nleaf in Nleaf_tab:
        liste_clusters = arbre_fonctions.construction_arbre(Npoints, Nleaf, points)
        feuille_tab.append(arbre_fonctions.nombre_feuilles(liste_clusters))

    feuille_theorique = [Npoints / Nleaf for Nleaf in Nleaf_tab]

    print("Nleaf_tab :", Nleaf_tab)
    print("feuille_tab :", feuille_tab)
    print("feuille_theorique :", feuille_theorique)

    # Tracé du graphique
    plt.figure(figsize=(8, 5))
    plt.plot(Nleaf_tab, feuille_tab, 'o-', label="Nombre de feuilles mesuré")
    plt.plot(
        Nleaf_tab, 
        feuille_theorique, 
        's--', 
        label=rf"Théorique : $N_{{\mathrm{{points}}}} / N_{{\mathrm{{leaf}}}}$ "
    )

    plt.xlabel(r"$N_{\mathrm{leaf}}$")
    plt.ylabel("Nombre de feuilles")
    plt.title(rf"Nombre de feuilles de l'arbre en fonction de $N_{{\mathrm{{leaf}}}}$ ($a = {a}$, $N_{{\mathrm{{points}}}} = {Npoints}$)")
    plt.grid(True)
    plt.legend()
    plt.show()
