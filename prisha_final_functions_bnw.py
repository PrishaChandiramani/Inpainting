
from PIL import Image
import numpy as np
from skimage.util import view_as_windows
import luciano_final_functions as lf
import time
import matplotlib.pyplot as plt

# Partie codée par Prisha

def calcul_dist(p, q, p_mask):
    """
    Calcule la distance au carré entre deux patchs en niveaux de gris en utilisant un masque pour ne considérer que les pixels connus.

    Parameters:
    p (numpy.ndarray): Un patch centré en un pixel de la frontière de la région cible.
    q (numpy.ndarray): Un patch de la région source.
    p_mask (numpy.ndarray): Masque pour déterminer les pixels connus du patch p.

    Returns:
    float: Distance au carré entre les deux patchs masqués, c'est à dire la somme des distances au carrés entre les patchs en ne considérant que les pixels connus.

    Raises:
    ValueError: Si les dimensions de p et q ne correspondent pas.
    """
    # Vérifie si les deux patchs ont la même taille
    if p.shape != q.shape:
        raise ValueError("Les deux patchs n'ont pas la même taille")
    
    # Applique le masque aux deux patchs pour ne ocnsidérer que les pixels connus
    p_masked = p*p_mask
    q_masked = q*p_mask

    # Convertit les patchs masqués en float64 pour éviter l'overflow
    p_masked = p_masked.astype(np.float64)
    q_masked = q_masked.astype(np.float64)

    # Calcule la différence entre les deux patchs masqués
    diff = p_masked - q_masked

    # Calcule la somme des carrés des différences
    sum = np.sum(diff ** 2)

    # Retourne la somme
    return sum


def choose_q(target_region_mask, front, p, p_mask, im, patch_size):
    D = {}
    margin = 70
    source_region_mask = np.logical_not(target_region_mask)
    
    #Define the limits of the target region
    target_indices = np.argwhere(target_region_mask)
    min_x, min_y = target_indices.min(axis=0)
    max_x, max_y = target_indices.max(axis=0)

    #Define the limits of the source region with a margin
    min_x = max(patch_size, min_x - margin)
    max_x = min(source_region_mask.shape[0] - patch_size, max_x + margin + patch_size)
    min_y = max(patch_size, min_y - margin)
    max_y = min(source_region_mask.shape[1] - patch_size, max_y + margin + patch_size)

    #Extract the source region from the image
    source_region = im[min_x:max_x, min_y:max_y]
    source_region_mask = source_region_mask[min_x:max_x, min_y:max_y]

    # Extract patches from the image and the mask
    patches = view_as_windows(source_region, (patch_size, patch_size))
    mask_patches = view_as_windows(source_region_mask, (patch_size, patch_size))
    
    # Flatten the patches for easier processing
    patches = patches.reshape(-1, patch_size, patch_size)
    mask_patches = mask_patches.reshape(-1, patch_size, patch_size)
    
    # Filter valid patches
    for idx, mask in enumerate(mask_patches):
        if np.all(mask):
            patch = patches[idx]
            d = calcul_dist(p, patch, p_mask)
            i, j = np.unravel_index(idx, (source_region_mask.shape[0] - patch_size + 1, source_region_mask.shape[1] - patch_size + 1))
            D[(i + min_x, j + min_y)] = d

    """
    # Parcourir les patchs possibles dans la région source
    for i in range(min_x,max_x-patch_size):
        for j in range(min_y,max_y-patch_size):
            q_mask = source_region_mask[i:i+patch_size,j:j+patch_size]
            valid_patch = np.all(q_mask)
            if valid_patch:
                q = im[i:i+patch_size,j:j+patch_size]
                d = calcul_dist(p, q, p_mask)
                D[(i,j)] = d
    """
    
    
    # Find the patch with the minimum distance
    minimum_D = min(D, key=D.get)  # Returns the key of the minimum value
    q_opt = im[minimum_D[0]:minimum_D[0] + patch_size, minimum_D[1]:minimum_D[1] + patch_size]

    return q_opt



def update_target_region_mask(target_region_mask, selected_pixel, patch_size,im):
    
    """
    Met à jour le masque de la région cible en marquant la zone autour du pixel sélectionné comme remplie.

    Parameters:
    target_region_mask (numpy.ndarray): Masque de la région cible.
    selected_pixel (tuple): Coordonnées du pixel sélectionné (x, y).
    patch_size (int): Taille du patch.
    im (numpy.ndarray): Image source.

    Returns:
    numpy.ndarray: Masque de la région cible mis à jour.
    """

    # Crée une copie du masque de la région cible
    updated_matrix = target_region_mask.copy()

    # Calcule la moitié de la taille du patch
    half_patch_size = patch_size // 2
   
    # Met à jour la zone autour du pixel sélectionné en la marquant comme remplie (False)
    updated_matrix [max(selected_pixel[0] - half_patch_size, 0): min(selected_pixel[0] + half_patch_size + 1, im.shape[0] - 1),max(selected_pixel[1] - half_patch_size, 0): min(selected_pixel[1] + half_patch_size + 1 , im.shape[1] - 1)]= np.array([[False for i in range (patch_size)] for j in range(patch_size)])
    
    # Retourne le masque mis à jour
    return updated_matrix


def neighbour_to_source_region(x, y, target_region_mask):

    """
    Vérifie si un pixel a des voisins dans la région source, s'il appartient à la frontière de la région cible.

    Parameters:
    x (int): Coordonnée x du pixel.
    y (int): Coordonnée y du pixel.
    target_region_mask (numpy.ndarray): Masque de la région cible.

    Returns:
    bool: True si le pixel a au moins un voisin dans la région source, False sinon.
    """

    # Crée le masque de la région source en inversant le masque de la région cible
    source_region_mask = 1. - target_region_mask

    # Initialise le compteur de voisins dans la région source
    number_of_source_region_neighours = 0
    # Vérifie si le pixel a des voisins dans la région source e nfonction de sa position
    if x > 0 and x < target_region_mask.shape[0] - 1:
        if y > 0 and y < target_region_mask.shape[1] - 1:
            # Pixel qui n'est pas sur les bords
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            # Pixel sur le bord gauche
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y + 1]
        else:
            # Pixel sur le bord droit
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y - 1]
    elif x == 0:
        # Pixel sur le bord supérieur
        if y > 0 and y < target_region_mask.shape[1] - 1:
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            # Pixel sur le coin supérieur gauche
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y + 1]
        else:
            # Pixel sur le coin supérieur droit
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y - 1]
    else:
        # Pixel sur le bord inférieur
        if y > 0 and y < target_region_mask.shape[1] - 1:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            # Pixel sur le coin inférieur gauche
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y + 1]
        else:
            # Pixel sur le coin inférieur droit
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y - 1]
    return number_of_source_region_neighours > 0



def front_detection(im, target_region_mask, half_patch_size):
    """
    Détecte la frontière de la région cible dans l'image.

    Parameters:
    im (numpy.ndarray): Image source.
    target_region_mask (numpy.ndarray): Masque de la région cible.
    half_patch_size (int): Moitié de la taille du patch.

    Returns:
    numpy.ndarray or str: Masque de la frontière ou un message indiquant qu'il n'y a pas de région cible.
    """

    # Vérifie si les dimensions de la région cible et de l'image correspondent
    if target_region_mask.shape != im.shape:
        raise ValueError('target_region_mask and im must have the same shape')
    
    # Vérifie s'il reste encore un pixel dans la région cible à modifier
    if np.all(target_region_mask == np.array([[False for i in range(im.shape[1])] for j in range(im.shape[0])])):
        return ("No target region")
    else : 
        # Initialise le masque du front avec des valeurs False
        front = np.array([[False for i in range(im.shape[1])] for j in range(im.shape[0])])
        
        # Parcourt chaque pixel de l'image en évitant les bords
        for x in range(half_patch_size, im.shape[0] - half_patch_size):
            for y in range(half_patch_size, im.shape[1] - half_patch_size):
                # Vérifie si le pixel fait partie de la région cible
                if target_region_mask[x, y]:
                    # Met à jour le masque du front si le pixel de la région cible a des voisins dans la région source, c'est à dire qu'il appartient à la frontière
                    front[x, y] = neighbour_to_source_region(x, y, target_region_mask)
        # Retourne le masque du front
        return front
    


def patch_search_compatible(target_region_mask, im, patch_size):

    """
    Applique la méthode d'inpainting pour remplir la région cible dans une image patch par patch jusqu'à ce qu'il n'y ait plus de pixels dans la région cible.

    Parameters:
    target_region_mask (numpy.ndarray): Masque de la région cible.
    im (numpy.ndarray): Image source.
    patch_size (int): Taille du patch.

    Returns:
    numpy.ndarray: Image avec la région cible remplie par la méthode d'inpainting.
    """

    # Convertit l'image en matrice numpy
    im_matrix = np.array(im)
    # Crée une copie de la matrice de l'image
    new_matrix = im_matrix.copy()

    # Remplit la région cible avec des pixels blancs (valeur 255)
    new_matrix = new_matrix * (1- target_region_mask) + target_region_mask * 255

    # Calcule la moitié de la taille du patch
    half_patch_size = patch_size // 2

    # Initialise la matrice de confiance
    confidence_matrix = 1. - np.copy(target_region_mask)
    
    # Tant qu'il reste des pixels dans la région cible à remplir
    while target_region_mask.any():
        # Détecte la frontière de la région cible
        front = front_detection(new_matrix, target_region_mask, patch_size // 2)
        
        # Sélectionne le pixel avec la priorité la plus élevée sur la frontière ainsi que sa confiance, son terme de données et sa priorité
        pixel, confidence, data_term, priority = lf.pixel_with_max_priority(front, new_matrix, im,target_region_mask, confidence_matrix, im.shape, patch_size)
        
        # Si le pixel fait partie de la région cible
        if target_region_mask[pixel[0],pixel[1]] == True:
            # Extrait le patch autour du pixel sélectionné
            patch = new_matrix[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)]
            
            # Extrait le masque du patch pour déterminer les valeurs connues et inconnues dans le patch sélectionné
            patch_mask = target_region_mask[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)]
            
            # Inverse le masque du patch pour obtenir les valeurs de la région source
            patch_mask = np.logical_not(patch_mask)
            
            #S électionne le patch q optimal de l'image source pour correspondre au patch p de la région cible (le patch le plus proche du patch p)
            q_patch = choose_q(target_region_mask, front, patch,  patch_mask, new_matrix, patch_size)
            
            # Met à jour la matrice de l'image à l'aide du patch q optimal en remplaçant les valeurs inconnues du patch considéré par les valeurs correspondantes dans le patch q optimal
            new_matrix [max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)] = q_patch*(1-patch_mask) + patch*patch_mask
            
            # Met à jour la matrice de confiance en mettant à jour les valeurs de confiance des pixels dans le patch considéré
            confidence_matrix = lf.update_confidence(confidence_matrix, target_region_mask, pixel, confidence, patch_size, im.shape)
            
            # Met à jour le masque de la région cible en marquant la zone autour du pixel sélectionné comme remplie
            target_region_mask = update_target_region_mask(target_region_mask, pixel, patch_size,new_matrix)
            
     # Convertit la matrice de l'image en uint8   
    new_matrix = new_matrix.astype(np.uint8)  
    
    # Retourne la matrice de l'image avec la région cible remplie      
    return new_matrix