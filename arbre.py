import fonctions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

#================== CLASS NOEUD =============

class Noeud:
    def __init__(self, box : list, list_pts : list, niveau : int, indice_local : int, fils1 : "Noeud" = None, fils2 : "Noeud" = None):
        self.box = box
        self.list_pts = list_pts
        self.niveau = niveau
        self.indice_local = indice_local
        self.fils1 = fils1
        self.fils2 = fils2
        return 

    @property
    def is_leaf(self):
        """Retourne True si le nœud est une feuille."""
        return self.fils1 is None and self.fils2 is None

    @property
    def n_points(self):
        """Nombre de degrés de liberté dans le cluster."""
        return len(self.list_pts)

#================== ARBRE =============

def construction_arbre(Npoints, Nleaf, points):

    def box_initial(points):     
        # initialisation avec le premier point de la liste
        x_min = x_max = points[0][0]
        y_min = y_max = points[0][1]
        
        for point in points:
            if point[0]>x_max : x_max = point[0]
            if point[0]<x_min : x_min = point[0]

            if point[1]>y_max : y_max = point[1]
            if point[1]<y_min : y_min = point[1]

        return [[x_min,y_min],[x_min,y_max],[x_max,y_min],[x_max,y_max]]

    def iteration(noeud):
        cluster.append(noeud)

        if len(noeud.list_pts) >= Nleaf:
            xmin = min([pts[0] for pts in noeud.box])
            ymin = min([pts[1] for pts in noeud.box])

            xmax = max([pts[0] for pts in noeud.box])
            ymax = max([pts[1] for pts in noeud.box])

            # Choix de la direction de découpe
            Lx = xmax - xmin
            Ly = ymax - ymin

            list_pts1, list_pts2 = [], []

            if Lx >= Ly: #on decoupe selon y
                #fils 1: xmin xmax/2
                #fils 2 : xmax/2 xmax
                #rien ne change selon y
                for pts in noeud.list_pts:
                    xmid = (xmax+xmin)/2
                    if pts[0] <= xmid:
                        list_pts1.append(pts)
                    else:
                        list_pts2.append(pts)

                # creation des points de la boîte
                box_1 = [(xmin, ymin), (xmid, ymin), (xmid, ymax), (xmin, ymax)]
                box_2 = [(xmid, ymin), (xmax, ymin), (xmax, ymax), (xmid, ymax)]

            else:
                for pts in noeud.list_pts:
                    ymid = (ymax+ymin)/2
                    if pts[1] <= ymid:
                        list_pts1.append(pts)
                    else:
                        list_pts2.append(pts)

                box_1 = [(xmin, ymin), (xmax, ymin), (xmax, ymid), (xmin, ymid)]
                box_2 = [(xmin, ymid), (xmax, ymid), (xmax, ymax), (xmin, ymax)]

            fils1 = Noeud(box_1, list_pts1, niveau=noeud.niveau+1, indice_local=0)
            fils2 = Noeud(box_2, list_pts2, niveau=noeud.niveau+1, indice_local=1)

            noeud.fils1 = fils1
            noeud.fils2 = fils2

            iteration(fils1)
            iteration(fils2)

    cluster = []
    racine = Noeud(box_initial(points), points, niveau=0, indice_local=0)
    iteration(racine)

    return cluster

#================== AFFICHAGE TERMINAL =============

def afficher_arbre_terminal(noeud, prefixe=""):
    """
    Affiche la structure hiérarchique de l'arbre dans la console.
    """
    if noeud is None:
        return

    # Détermine l'étiquette selon qu'il s'agisse de la racine ou d'un fils
    if noeud.niveau == 0:
        etiquette = "[Racine]"
    else:
        etiquette = f"[Fils {noeud.indice_local}]"

    # Récupération des dimensions réelles de la boîte du nœud
    xmin = min(p[0] for p in noeud.box)
    xmax = max(p[0] for p in noeud.box)
    ymin = min(p[1] for p in noeud.box)
    ymax = max(p[1] for p in noeud.box)

    # Indicateur si le nœud est une feuille
    statut = "🍃 Feuille" if noeud.is_leaf else "🔀 Nœud interne"

    # Affichage des informations du nœud
    print(
        f"{prefixe}{etiquette} Niv {noeud.niveau} | "
        f"X: [{xmin:.2f}, {xmax:.2f}] | Y: [{ymin:.2f}, {ymax:.2f}] | "
        f"{noeud.n_points} pts | {statut}"
    )

    # Préparation du décalage pour la profondeur suivante
    nouvel_indentation = prefixe + "    └── "

    # Appels récursifs sur les enfants
    if noeud.fils1:
        afficher_arbre_terminal(noeud.fils1, nouvel_indentation)
    if noeud.fils2:
        afficher_arbre_terminal(noeud.fils2, nouvel_indentation)

#================== AFFICHAGE GEOMETRIE =============
def affichage_clusters(points, segments, liste_clusters, Npoints, Nleaf, a, afficher_ids=True):
    """
    Affiche le maillage et les clusters avec un panneau d'information des paramètres sur le côté.
    
    Parameters:
    -----------
    points : np.ndarray
        Coordonnées des points (N, 2).
    segments : list / np.ndarray
        Paires d'indices des segments.
    liste_clusters : list de Noeud (ou Noeud racine)
        Liste des nœuds de l'arbre.
    Npoints : int
        Nombre total de points du maillage.
    Nleaf : int
        Nombre maximal de points par feuille.
    a : float/int
        Paramètre géométrique (ex: rayon ou demi-grand axe).
    afficher_ids : bool
        Si True, affiche les infos sur les feuilles.
    """
    X = points[:, 0]
    Y = points[:, 1]

    # Création d'une figure avec 2 zones : [Graphique principal (4/5), Panneau latéral (1/5)]
    fig, (ax_graph, ax_info) = plt.subplots(
        1, 2, figsize=(14, 10),
        gridspec_kw={'width_ratios': [4, 1]}
    )

    # -------------------------------------------------------------
    # 1. ZONE GRAPHIQUE PRINCIPALE (ax_graph)
    # -------------------------------------------------------------
    # Tracé des segments du maillage
    for noeud1, noeud2 in segments:
        ax_graph.plot([X[noeud1], X[noeud2]], [Y[noeud1], Y[noeud2]], color='blue', alpha=0.4, zorder=1)

    # Tracé des points
    ax_graph.scatter(X, Y, color='red', s=20, zorder=3, label="Points")

    # Reconstitution de la liste des nœuds si la racine est passée seule
    if not isinstance(liste_clusters, list):
        nodes_to_visit = [liste_clusters]
        all_nodes = []
        while nodes_to_visit:
            curr = nodes_to_visit.pop()
            all_nodes.append(curr)
            if curr.fils1: nodes_to_visit.append(curr.fils1)
            if curr.fils2: nodes_to_visit.append(curr.fils2)
        liste_clusters = all_nodes

    # Tracé des boîtes de chaque cluster
    for noeud in liste_clusters:
        xmin = min(p[0] for p in noeud.box)
        xmax = max(p[0] for p in noeud.box)
        ymin = min(p[1] for p in noeud.box)
        ymax = max(p[1] for p in noeud.box)

        largeur = xmax - xmin
        hauteur = ymax - ymin

        if noeud.is_leaf:
            # Boîtes des feuilles (vertes)
            rect = patches.Rectangle(
                (xmin, ymin), largeur, hauteur,
                linewidth=1.5, edgecolor='#2ea44f', facecolor='#2ea44f', alpha=0.12, zorder=2
            )
            ax_graph.add_patch(rect)

            rect_border = patches.Rectangle(
                (xmin, ymin), largeur, hauteur,
                linewidth=1.2, edgecolor='#2ea44f', facecolor='none', zorder=4
            )
            ax_graph.add_patch(rect_border)

            if afficher_ids:
                ax_graph.text(
                    xmin + largeur / 2, ymin + hauteur / 2,
                    f"Niv {noeud.niveau}\n({noeud.n_points} p)",
                    color='#1a5e2c', fontsize=8, ha='center', va='center',
                    fontweight='bold', zorder=5
                )
        else:
            # Boîtes des nœuds internes (pointillés orange)
            rect = patches.Rectangle(
                (xmin, ymin), largeur, hauteur,
                linewidth=1, edgecolor='#d97706', linestyle='--', facecolor='none', zorder=2
            )
            ax_graph.add_patch(rect)

    # Configuration des axes
    ax_graph.set_aspect('equal', adjustable='box')
    marge = 0.2
    ax_graph.set_xlim(np.min(X) - marge, np.max(X) + marge)
    ax_graph.set_ylim(np.min(Y) - marge, np.max(Y) + marge)
    ax_graph.grid(True, linestyle='--', alpha=0.4)
    ax_graph.set_title("Partitionnement par Arbre Binaire (Clusters)", fontsize=13)

    # Légende du graphique
    legend_elements = [
        Line2D([0], [0], color='blue', lw=1, alpha=0.6, label='Segments'),
        Line2D([0], [0], marker='o', color='w', label='Points', markerfacecolor='red', markersize=6),
        patches.Patch(facecolor='#2ea44f', edgecolor='#2ea44f', alpha=0.3, label='Feuille'),
        Line2D([0], [0], color='#d97706', linestyle='--', lw=1, label='Nœud interne')
    ]
    ax_graph.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # -------------------------------------------------------------
    # 2. PANNEAU LATÉRAL D'INFORMATIONS (ax_info)
    # -------------------------------------------------------------
    ax_info.axis('off')  # On masque les axes pour le panneau d'affichage texte

    # Encadré visuel pour les paramètres
    p_box = patches.FancyBboxPatch(
        (0.05, 0.50), 0.90, 0.45,
        boxstyle="round,pad=0.03,rounding_size=0.03",
        facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.5
    )
    ax_info.add_patch(p_box)

    # Titre du panneau
    ax_info.text(
        0.5, 0.91, "PARAMÈTRES",
        fontsize=12, fontweight='bold', ha='center', va='top', color='#1e293b'
    )

    # Ligne de séparation sous le titre
    ax_info.plot([0.15, 0.85], [0.87, 0.87], color='#cbd5e1', lw=1)

    # Contenu texte des paramètres
    texte_parametres = (
        f"• Npoints : {Npoints}\n\n"
        f"• Nleaf   : {Nleaf}\n\n"
        f"• a       : {a}"
    )

    ax_info.text(
        0.15, 0.82, texte_parametres,
        fontsize=11, fontweight='semibold', color='#334155',
        va='top', linespacing=1.4
    )

    plt.tight_layout()
    plt.show()

