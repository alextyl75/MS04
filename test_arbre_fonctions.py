import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


#================== Test de validité =============

def test_validite(arbre, points_maillage):
    """
    On parcourt l'arbre, on colorie les noeuds sur une feuille. Pour que l'arbre soit valide, il faut que :
- tous les points soient coloriés exactement 1 fois (test 1 et 2)
- points_colorie == points_maillage à la fin du code (test 3)
    """

    points_colories = []

    def coloriage(noeud):
        if noeud.is_leaf:
            # Extraire uniquement les coordonnées pour la vérification avec points_maillage
            points_colories.extend([pt[1] for pt in noeud.list_pts])
        else:
            coloriage(noeud.fils1)
            coloriage(noeud.fils2)

    coloriage(arbre[0])

    # premier test : taille des listes
    if len(points_colories) != len(points_maillage):
        print(f"Arbre non-valide : {len(points_colories)} points clusters vs {len(points_maillage)} points maillage.")
        return False

    # verification qu'il n'y a pas de doublon (conversion en set qui ne peut pas contenir de doublons)
    # conversion en tuples pour pouvoir convertir en set
    points_colories_tuples = [tuple(p) for p in points_colories]
    points_colories_set = set(points_colories_tuples)

    # deuxieme test : doublons
    if len(points_colories_set) != len(points_colories):
        print(f"Abre non-valide : {len(points_colories) - len(points_colories_set)} doublons.")
        return False

    # troisieme test : comparaison des ensembles de points
    points_maillage_tuples = [tuple(p) for p in points_maillage]
    points_maillage_set = set(points_maillage_tuples)

    if points_colories_set != points_maillage_set:
        print("Arbre non-valide : les points de l'arbre ne correspondent pas aux points du maillage.")
        return False

    print("Arbre valide !")
    return True

# Test de permutation

def permutation_indice(arbre):
    sigma = []
    
    def extraire_indices(noeud):
        if noeud is None:
            return
        if noeud.is_leaf:
            sigma.extend([pt[0] for pt in noeud.list_pts])
        else:
            extraire_indices(noeud.fils1)
            extraire_indices(noeud.fils2)

    racine = arbre[0] if isinstance(arbre, list) else arbre
    extraire_indices(racine)

    return sigma
    
def test_permutation(arbre, points_maillage):
    """
    Parcourt l'arbre, extrait l'ordre des indices d'origine (permutation)
    définis par le parcours des feuilles, et vérifie que sigma est bien une 
    permutation de {0, ..., Npoints - 1}.
    """
    sigma = permutation_indice(arbre)

    Npoints = len(points_maillage)

    # Vérification de la taille
    if len(sigma) != Npoints:
        print(f"Arbre non-valide : la permutation contient {len(sigma)} éléments vs {Npoints} attendus.")
        return False

    # Conversion en set pour les comparaisons
    sigma_set = set(sigma)

    if len(sigma_set) != len(sigma):
        print(f"Arbre non-valide : {len(sigma) - len(sigma_set)} doublons.")
        return False

    # Test 3 : vérification de l'ensemble exact {0, 1, ..., Npoints - 1}
    indices_attendus = set(range(Npoints))

    if sigma_set != indices_attendus:
        print("Arbre non-valide : la liste des indices ne correspond pas à {0, ..., N-1}.")
        return False

    print("Permutation valide : sigma est bien une permutation de {0, ..., N-1} !")
    return True

def verification_indices_consecutifs(arbre, points_maillage):
    if test_permutation(arbre, points_maillage):
        sigma = permutation_indice(arbre)

        # Création du dictionnaire inverse 
        sigma_inv_dict = {ancien_id: nouvel_id for nouvel_id, ancien_id in enumerate(sigma)}
        print("sigma_inv_dict", sigma_inv_dict  )

        def test_index(noeud):
            # Nouveaux indices attribués par sigma à cette feuille
            sigma_inv_tab = [sigma_inv_dict[id] for id, _ in noeud.list_pts]

            # Sécurité si la feuille ne contient aucun point
            if len(sigma_inv_tab) == 0:
                return True

            # Test de la taille de la plage 
            if len(sigma_inv_tab) != (max(sigma_inv_tab) - min(sigma_inv_tab) + 1):
                print(f"Mauvaise longueur sur la feuille : {len(sigma_inv_tab)} pts vs plage [{min(sigma_inv_tab)}, {max(sigma_inv_tab)}]")
                return False
            
            # Test : trier et vérifier qu'il n'y a aucun trou
            sigma_inv_tab_trie = sorted(sigma_inv_tab)
            for i in range(len(sigma_inv_tab_trie) - 1):
                if sigma_inv_tab_trie[i] + 1 != sigma_inv_tab_trie[i+1]:
                    print("Trou détecté dans la suite des nouveaux indices !")
                    return False
                
            return True
       
        def parcourt(noeud):
            if noeud is None:
                return True
            if noeud.is_leaf:
                return test_index(noeud)
            else:
                # Propage le résultat booléen des deux fils
                return parcourt(noeud.fils1) and parcourt(noeud.fils2)

        # Récupération de la racine 
        racine = arbre[0] if isinstance(arbre, list) else arbre
        
        valide = parcourt(racine)
        if valide:
            print("Tous les clusters contiennent des indices consécutifs dans la nouvelle numérotation !")
        return valide

    return False
