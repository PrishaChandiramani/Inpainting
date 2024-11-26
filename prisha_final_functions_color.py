from PIL import Image
import numpy as np
from skimage.util import view_as_windows
import luciano_final_functions as lf
import time
from scipy import signal
import matplotlib.pyplot as plt

# Partie codée par Prisha


def calcul_dist_color(p,q, p_mask):

    """
    Calcule la distance entre deux patchs de couleur en utilisant un masque pour ne considérer que les pixels connus.

    Parameters:
    p (numpy.ndarray): Un patch de couleur centré en un pixel de la frontière de la région cible, un tableau numpy 3D.
    q (numpy.ndarray): Un patch de couleur de la région source, un tableau numpy 3D.
    p_mask (numpy.ndarray): Le masque à appliquer aux deux patchs, qui représente l'ensemble des pixels connus du patch p. 

    Returns:
    float: La somme des différences au carré entre les patchs masqués, c'est à dire la somme des distances au carrés entre les patchs en ne considérant que les pixels connus.

    Raises:
    ValueError: Si les dimensions de p et q ne correspondent pas.
    """
   
    # Vérifier que les deux patchs ont la même taille
    if (p.shape[0],p.shape[1]) != (q.shape[0],q.shape[1]):
        print("p.shape:",p.shape)
        print("q.shape:",q.shape)
        raise ValueError("Les deux patchs n'ont pas la même taille")
    
    # Appliquer le masque aux deux patchs pour ne considérer que les pixels connus
    p_masked = p*p_mask
    q_masked = q*p_mask

    # Convertir en float pour éviter l'overflow
    p_masked = p_masked.astype(np.float64)
    q_masked = q_masked.astype(np.float64)

    # Calculer la différence entre les deux patchs masqués
    diff = p_masked - q_masked

    # Calculer la somme des différences au carré
    sum_squared_diff = np.sum(diff ** 2, axis=(0,1))

    # Retourner la somme des différences au carré
    return np.sum(sum_squared_diff)


def choose_q(target_region_mask, p, p_mask, im, patch_size,margin = 60):

    """
    Choisit le patch q optimal (le plus proche du patch p) de l'image source pour correspondre au patch p de la région cible.

    Parameters:
    target_region_mask (numpy.ndarray): Masque de la région cible.
    p (numpy.ndarray): Patch de la région cible.
    p_mask (numpy.ndarray): Masque appliqué au patch p.
    im (numpy.ndarray): Image source.
    patch_size (int): Taille du patch.
    margin (int, optional): Marge autour de la région cible pour définir la région source. Par défaut à 60.

    Returns:
    numpy.ndarray: Le patch q optimal de l'image source.
    """
    
    D = {} # Dictionnaire pour stocker les distances calculées
    
    # Crée un masque de la région source en inversant le masque de la région cible
    source_region_mask = np.logical_not(target_region_mask)
    source_region_mask_size = source_region_mask.shape
    

    
    # Définir les limites de la région cible
    target_indices = np.argwhere(target_region_mask)
    min_x, min_y,min_z = target_indices.min(axis=0)
    max_x, max_y,min_z = target_indices.max(axis=0)
    

    # Définir les limites de la région source avec une marge
    min_x = max(0,min_x - margin)
    max_x = min(source_region_mask_size[0]- patch_size,max_x + margin + patch_size)
    min_y = max(0,min_y - margin)
    max_y = min(source_region_mask_size[1]-patch_size,max_y + margin + patch_size)

    # Parcourir les patchs possibles dans la région source
    for i in range(min_x,max_x-patch_size):
        for j in range(min_y,max_y-patch_size):
            q_mask = source_region_mask[i:i+patch_size,j:j+patch_size]
            valid_patch = np.all(q_mask)
            if valid_patch:
                q = im[i:i+patch_size,j:j+patch_size]
                d = calcul_dist_color(p, q, p_mask)
                D[(i,j)] = d
    
    # Trouve le patch avec la distance minimale
    minimum_D = min(D, key=D.get)  # Retourne la clé de la valeur minimale
    q_opt = im[minimum_D[0]:minimum_D[0] + patch_size, minimum_D[1]:minimum_D[1] + patch_size]
    
    # Retourne le patch optimal
    return q_opt



def update_target_region_mask(target_region_mask, selected_pixel, patch_size,im_size):
    """
    Met à jour le masque de la région cible en marquant une zone autour du pixel sélectionné comme traitée et ne faisant plus partie de la région cible mais de la région source .

    Parameters:
    target_region_mask (numpy.ndarray): Masque de la région cible.
    selected_pixel (tuple): Coordonnées du pixel sélectionné (x, y).
    patch_size (int): Taille du patch.
    im_size (tuple): Taille de l'image (hauteur, largeur).

    Returns:
    numpy.ndarray: Masque de la région cible mis à jour.
    """
    
    # Crée une copie du masque de la région cible pour la mise à jour
    updated_matrix = target_region_mask.copy()

    # Calcule la moitié de la taille du patch
    half_patch_size = patch_size // 2
    # Mise à jour de la région cible autour du pixel sélectionné
    updated_matrix [max(selected_pixel[0] - half_patch_size, 0): min(selected_pixel[0] + half_patch_size + 1, im_size[0] - 1),max(selected_pixel[1] - half_patch_size, 0): min(selected_pixel[1] + half_patch_size + 1 , im_size[1] - 1)]= np.array([[[False] for i in range (patch_size)] for j in range(patch_size)])
    # Retourne le masque de la région cible mis à jour
    return updated_matrix


def neighbour_to_source_region(x, y, target_region_mask):

    """
    Vérifie si un pixel (x, y) de la région cible a des voisins dans la région source, autrement dit s'il appartient à la frontière de la région cible.

    Parameters:
    x (int): Coordonnée x du pixel.
    y (int): Coordonnée y du pixel.
    target_region_mask (numpy.ndarray): Masque de la région cible.

    Returns:
    bool: True si le pixel a au moins un voisin dans la région source, False sinon.
    """

    # Crée un masque de la région source en inversant le masque de la région cible
    source_region_mask = ~target_region_mask
    number_of_source_region_neighours = 0
    target_region_mask_size = target_region_mask.shape

    # Vérifie si le pixel a des voisins dans la région source en fonction de sa position
    if x > 0 and x < target_region_mask_size[0] - 1:
        if y > 0 and y < target_region_mask_size[1] - 1:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0] + source_region_mask[x, y + 1,0]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x + 1, y,0] + source_region_mask[x, y + 1,0]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0]
    elif x == 0:
        if y > 0 and y < target_region_mask_size[1] - 1:
            number_of_source_region_neighours += source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0] + source_region_mask[x, y + 1,0]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x + 1, y,0] + source_region_mask[x, y + 1,0]
        else:
            number_of_source_region_neighours += source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0]
    else:
        if y > 0 and y < target_region_mask_size[1] - 1:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x, y - 1,0] + source_region_mask[x, y + 1,0]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x, y + 1,0]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x, y - 1,0]
    # Retourne True si le pixel a au moins un voisin dans la région source, sinon False
    return number_of_source_region_neighours > 0



def front_detection(im, target_region_mask,half_patch_size,im_size):
    """
    Détecte la frontière entre la région cible et la région source dans une image.

    Parameters:
    im (numpy.ndarray): Image source.
    target_region_mask (numpy.ndarray): Masque de la région cible.
    half_patch_size (int): Moitié de la taille du patch.
    im_size (tuple): Taille de l'image (hauteur, largeur).

    Returns:
    numpy.ndarray or str: Masque de la frontière ou un message indiquant qu'il n'y a pas de région cible.
    """
   
    # Vérifie s'il reste encore un pixel dans la région cible
    if np.all(target_region_mask == np.array([[[False] for i in range(im_size[1])] for j in range(im_size[0])])):
        return ("No target region")
    else : 
        # Initialise le masque de la frontière avec des valeurs False
        front = np.array([[False for i in range(im_size[1])] for j in range(im_size[0])])

        # Parcourt chaque pixel de l'image en évitant les bords
        for x in range(half_patch_size,im_size[0]-half_patch_size):
            for y in range(half_patch_size,im_size[1]-half_patch_size):
                # Vérifie si le pixel appartient à la région cible
                if target_region_mask[x, y,0]:
                    # Détermine si le pixel a des voisins dans la région source
                    front[x, y] = neighbour_to_source_region(x, y, target_region_mask)
        # Retourne le masque de la frontière
        return front
  

def patch_search_compatible(target_region_mask, im, patch_size,margin = 70):

    """
    Applique la méthode d'inpainting pour remplir la région cible dans une image patch par patch jusqu'à ce qu'il n'y ait plus de pixels dans la région cible.

    Parameters:
    target_region_mask (numpy.ndarray): Masque de la région cible.
    im (numpy.ndarray): Image source.
    patch_size (int): Taille du patch.
    margin (int, optional): Marge pour définir une plus petite région source autour de la région cible pour la recherche de patch. Par défaut à 70.

    Returns:
    numpy.ndarray: Nouvelle image avec la région cible remplie.
    """

    # Convertit l'image en une matrice numpy et crée une copie
    im_matrix = np.array(im)
    new_matrix = im_matrix.copy()
    im_size = im.shape # Garde en mémoire la taille de l'image
   
    half_patch_size = patch_size // 2 # Calcule la moitié de la taille du patch

    # Initialise la matrice de confiance
    confidence_matrix = 1. - np.copy(target_region_mask)
    while target_region_mask.any():
        # Détecte la frontière entre la région cible et la région source
        front = front_detection(new_matrix, target_region_mask,half_patch_size,im_size)
        
        # Sélectionne le pixel avec la priorité la plus élevée sur la frontière ainsi que sa confiance, son terme de données et sa priorité
        pixel, confidence, data_term, priority = lf.pixel_with_max_priority(front, new_matrix, im,target_region_mask, confidence_matrix, im_size, patch_size)
        
        # Vérifie si le pixel sélectionné appartient toujours à la région cible
        if target_region_mask[pixel[0],pixel[1]] == np.array([True]):
            # Sélectionne le patch de la région cible et son masque centré sur le pixel sélectionné avec la priorité la plus élevée, il s'agit de la zone de la région cible que nous allons remplir
            patch = new_matrix[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im_size[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im_size[1] - 1)]
            
            # Extrait le masque du patch correspondant
            patch_mask = target_region_mask[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im_size[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im_size[1] - 1)]
            
            # Inverse le masque du patch pour obtenir les valeurs de la région source
            patch_mask = np.logical_not(patch_mask)
           
            # Sélectionne le patch q optimal de l'image source pour correspondre au patch p de la région cible (le patch le plus proche du patch p)
            q_patch = choose_q(target_region_mask, patch,  patch_mask, new_matrix, patch_size,margin)
        
            # Met à jour la matrice de l'image à l'aide du patch q optimal en remplaçant les valeurs inconnues du patch considéré par les valeurs correspondantes dans le patch q optimal
            new_matrix [max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im_size[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im_size[1] - 1)] = q_patch*(1-patch_mask) + patch*patch_mask
          
            # Met à jour la matrice de confiance
            confidence_matrix = lf.update_confidence(confidence_matrix, target_region_mask, pixel, confidence, patch_size, im_size)
            
            # Met à jour le masque de la région cible
            target_region_mask = update_target_region_mask(target_region_mask, pixel, patch_size,im_size)
             
    # Convertit la nouvelle matrice en type uint8
    new_matrix = new_matrix.astype(np.uint8) 

    # Retourne la nouvelle image avec la région cible remplie       
    return new_matrix