import fonctions
import numpy as np
import matplotlib.pyplot as plt

# Parametres
Npoints = 100
Nleaf = 10

class noeud:
    def __init__(self, min, max, niveau, indice_local, fils1, fils2):
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
        return len(self.indices)

def construction_cluster(Npoints, Nleaf, distance):

    def generation_noeud(noeud):


    