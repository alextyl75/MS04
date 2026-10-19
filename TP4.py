import fonctions
import numpy as np
import matplotlib.pyplot as plt

# Parametres
Npoints = 100
Nleaf = 10

class Noeud:
    def __init__(self, min, max, niveau : int, indice_local : int, fils1 : "Noeud" = None, fils2 : "Noeud" = None):
        self.min = min
        self.max = max
        self.niveau = niveau
        self.indice_local = indice_local
        self.fils1 = fils1
        self.fils2 = fils2
        return 

    @property
    def index_tab(self):
        """Renvoie le tableau d'indice"""
        return [i for i in range(self.min, self.max+1)]

    @property
    def is_leaf(self):
        """Retourne True si le nœud est une feuille."""
        return self.fils1 is None and self.fils2 is None

    @property
    def n_points(self):
        """Nombre de degrés de liberté dans le cluster."""
        return len(self.index_tab)

def construction_arbre(Npoints, Nleaf,):

    def iteration(noeud):
        cluster.append(noeud)
        
        if (noeud.max - noeud.min) >= Nleaf:
            list_points = noeud.index_tab

            list_fils1 = list_points[:(len(list_points)//2)]
            fils1 = Noeud(min=min(list_fils1), max=max(list_fils1), niveau = noeud.niveau+1, indice_local=0)

            list_fils2 = list_points[(len(list_points)//2):]
            fils2 = Noeud(min=min(list_fils2), max=max(list_fils2), niveau = noeud.niveau+1, indice_local=1)

            noeud.fils1 = fils1
            noeud.fils2 = fils2

            iteration(fils1)
            iteration(fils2)

    cluster = []
    racine = Noeud(min=0, max=Npoints-1, niveau=0, indice_local=0)
    iteration(racine)

    return cluster

def afficher_arbre(noeud, prefixe=""):
    """Affiche l'arbre de manière hiérarchique dans le terminal."""
    if noeud is None:
        return
    
    # Détermine le symbole selon le fils (0 ou 1)
    type_fils = f"[Fils {noeud.indice_local}] " if noeud.niveau > 0 else "[Racine] "
    
    # Affiche les informations du nœud actuel
    print(f"{prefixe}{type_fils}Niveau {noeud.niveau} : Indices [{noeud.min} à {noeud.max}] ({noeud.n_points} points)")
    
    # Ajoute un décalage visuel pour les enfants
    nouvel_indentation = prefixe + "    └── "
    
    # Appel récursif sur les fils
    if noeud.fils1:
        afficher_arbre(noeud.fils1, nouvel_indentation)
    if noeud.fils2:
        afficher_arbre(noeud.fils2, nouvel_indentation)

# 1. Construction
liste_clusters = construction_arbre(Npoints, Nleaf)
noeud_racine = liste_clusters[0]

# 2. Affichage textuel dans le terminal
print("--- STRUCTURE DE L'ARBRE ---")
afficher_arbre(noeud_racine)

    







    