import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal
from new_gradient_final import new_gradient, new_orthogonal_front_vector

# Partie codée par Luciano


def priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, new_image):
    """ 
    Calcule la priorité d'un pixel comme le produit entre : 
    - Le terme de confiance : représente la confiance moyenne des pixels dans un patch centré autour du pixel. 
    - Le terme de données : basé sur le gradient au pixel et la géométrie de la frontière à restaurer. 
    Arguments: 
    - pixel (tuple ou list de deux entiers) : Coordonnées (x, y) du pixel à traiter. target_region_mask (numpy.ndarray) : Masque binaire indiquant la région cible à restaurer (1 pour les pixels à remplir, 0 sinon). 
    - confidence_matrix (numpy.ndarray) : Matrice de confiance des pixels, avec des valeurs entre 0 et 1. 
    - patch_size (int) : Taille d'un patch carré centré autour du pixel. 
    - image_size (tuple ou list de deux entiers) : Dimensions de l'image (largeur, hauteur) pour gérer les bordures. 
    - new_image (numpy.ndarray) : Image mise à jour utilisée pour calculer les gradients et la priorité. 
    Valeurs de retour: 
    tuple : 
    - confidence (float) : Terme de confiance pour le pixel (moyenne de la matrice de confiance du patch). 
    - data_term (float) : Terme de données, basé sur le gradient et la géométrie de la frontière. 
    - priority (float) : Produit du terme de confiance et du terme de données. 
    """
 
    confidence = 0.
    data_term = 0.

    # Calcul du terme de confiance
    
    pixel_x, pixel_y = pixel[0], pixel[1]
    half_patch_size = patch_size // 2

    # Coordonnées des coins du patch autour du pixel considéré
    xmin, xmax = max(pixel_x - half_patch_size, 0), min(pixel_x + half_patch_size + 1, image_size[0] - 1)
    ymin, ymax = max(pixel_y - half_patch_size, 0), min(pixel_y + half_patch_size + 1, image_size[1] - 1)
    
    mat = confidence_matrix[xmin:xmax, ymin:ymax] # Matrice de confiance des pixels autour du pixel considéré
    confidence = np.sum(mat) / ((xmax - xmin + 1) * (ymax - ymin + 1)) # Calcul de la moyenne de confiance de la matrice

    
    # Calcul du terme de données
    
    gradient = new_gradient(pixel, new_image, target_region_mask) # Calcul du gradient au pixel considéré
    orthogonal_to_gradient = [- gradient[1], gradient[0]] # Calcul du vecteur orthogonal au gradient
    front_orthogonal_vector = new_orthogonal_front_vector(pixel, target_region_mask) # Calcul du vecteur normal à la frontière
    data_term = np.abs(orthogonal_to_gradient[0] * front_orthogonal_vector[0] + orthogonal_to_gradient[1] * front_orthogonal_vector[1])
    
    data_term /= 255 # On normalise le terme de données pour avoir une priorité entre 0 et 1

    return confidence, data_term, confidence*data_term


def update_confidence(confidence_matrix, target_region_mask, selected_pixel, selected_pixel_confidence, patch_size, image_size):
    """ 
    Met à jour la matrice de confiance après le remplissage d'un patch centré autour d'un pixel sélectionné. Cette fonction calcule les nouvelles valeurs de confiance pour les pixels reconstruits dans un patch centré sur `selected_pixel`. Les nouvelles valeurs sont déterminées par la confiance associée au pixel sélectionné. 
    Arguments: 
    - confidence_matrix (numpy.ndarray) : Matrice de confiance actuelle, avec des valeurs entre 0 et 1. 
    - target_region_mask (numpy.ndarray) : Masque binaire indiquant la région cible à restaurer (1 pour les pixels à remplir, 0 sinon). 
    - selected_pixel (tuple ou list de deux entiers) : Coordonnées (x, y) du pixel sélectionné pour le remplissage. 
    - selected_pixel_confidence (float) : Confiance associée au pixel sélectionné, utilisée pour actualiser la confiance des pixels reconstruits dans le patch. 
    - patch_size (int) : Taille d'un patch carré centré autour du pixel. 
    - image_size (tuple ou list de deux entiers) : Dimensions de l'image (largeur, hauteur) pour gérer les bordures. 
    Valeur de retour: 
    - numpy.ndarray : Nouvelle matrice de confiance mise à jour après le remplissage du patch. 
    """

    new_confidence_matrix = np.copy(confidence_matrix) # Copie de l'ancienne matrice
    half_patch_size = patch_size // 2
    # Calcul des coordonnées du patch autour du pixel choisi
    xmin = max(selected_pixel[0] - half_patch_size, 0)
    xmax = min(selected_pixel[0] + half_patch_size + 1, image_size[0] - 1)
    ymin = max(selected_pixel[1] - half_patch_size, 0)
    ymax = min(selected_pixel[1] + half_patch_size + 1, image_size[1] - 1)
    
    confidence_patch = (1 - target_region_mask[xmin:xmax, ymin:ymax]) * confidence_matrix[xmin:xmax, ymin:ymax] # Anciennes valeurs de confiance sur le patch
    new_confidence_patch = target_region_mask[xmin:xmax, ymin:ymax] * selected_pixel_confidence + confidence_patch # Ajout des nouvelles valeurs de confiance sur le patch
    new_confidence_matrix[xmin:xmax, ymin:ymax] = new_confidence_patch # Ajout du nouveau patch sur la nouvelle matrice de confiance
    return new_confidence_matrix

def compute_gradient(img, boundary_mode='wrap'):
    """ 
    Calcule les gradients en x et en y pour tous les pixels de l'image donnée, en utilisant les noyaux de Sobel, et renvoie les résultats sous forme de tableau numpy de dimensions (image_height, image_width, 2). 
    Arguments: 
    - img (numpy.ndarray) : Image en niveaux de gris pour laquelle les gradients doivent être calculés. 
    - boundary_mode (str, optionnel) : Mode de gestion des bordures utilisé pour la convolution. Les options incluent 'wrap', 'constant', 'reflect', etc. (par défaut : 'wrap'). 
    Valeur de retour: 
    - numpy.ndarray : Tableau 3D contenant les gradients calculés : - La dimension [:, :, 0] correspond au gradient en x. - La dimension [:, :, 1] correspond au gradient en y. 
    """

    gradient_matrix = np.zeros((img.shape[0], img.shape[1], 2)) # On crée le array qui enregistrera les résultats
    # On définit les noyaux de Sobel en x et en y :
    gradient_core_x = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_core_y = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # On utilise la fonction convolve2d de Numpy pour calculer la convolution
    gradient_matrix[:, :, 0] = signal.convolve2d(img, gradient_core_x, mode='same', boundary=boundary_mode)
    gradient_matrix[:, :, 1] = signal.convolve2d(img, gradient_core_y, mode='same', boundary=boundary_mode)
    
    return gradient_matrix

def front_orthogonal_vectors(target_region_mask):
    """ 
    Calcule les vecteurs orthogonaux à la frontière d'une région cible définie par un masque binaire. Cette fonction utilise le gradient du masque de la région cible pour déterminer les vecteurs orthogonaux à la frontière. Les résultats sont normalisés pour obtenir des vecteurs unitaires. 
    Arguments: 
    - target_region_mask (numpy.ndarray) : Masque binaire indiquant la région cible (1 pour les pixels à remplir, 0 pour les autres). 
    Valeur de retour: 
    - numpy.ndarray : Tableau 3D de dimensions (image_height, image_width, 2) contenant les vecteurs orthogonaux à la frontière pour chaque pixel : 
        - La dimension [:, :, 0] correspond à la composante x. 
        - La dimension [:, :, 1] correspond à la composante y. 
    """

    front_orthogonal_vectors = np.zeros((target_region_mask.shape[0], target_region_mask.shape[1], 2)) # On crée le array qui enregistrera les résultats
    mask_gradient = compute_gradient(target_region_mask) # On appelle la fonction précédente compute_gradient()
    front_orthogonal_vectors = mask_gradient / np.abs(mask_gradient) # On normalise le résultat
    return front_orthogonal_vectors

def list_front_pixels(front_pixels_mask):
    """ 
    Renvoie la liste des pixels appartenant à la frontière de la région cible définie par un masque binaire. Cette fonction extrait les coordonnées (x, y) de tous les pixels non nuls dans le masque de la frontière de la région cible. 
    Arguments: 
    - front_pixels_mask (numpy.ndarray) : Masque binaire indiquant la frontière de la région cible (1 pour les pixels de la frontière, 0 sinon). 
    Valeur de retour: 
    - list : Liste de couples de coordonnées (x, y) des pixels de la frontière. 
    """

    non_null_indices = np.nonzero(front_pixels_mask) # Paire de listes d'indices du masque de la frontière de la région cible
    non_null_indices_list = [[non_null_indices[0][i], non_null_indices[1][i]] for i in range(len(non_null_indices[0]))] # Reconstruction d'une liste de couples de coordonnées
    return non_null_indices_list


def pixel_with_max_priority(front_pixels_mask, new_image, original_image, target_region_mask, confidence_matrix, image_size, patch_size):
    """ 
    Identifie le pixel de la frontière ayant la priorité maximale et renvoie ses informations. Cette fonction parcourt tous les pixels de la frontière de la région cible pour calculer leur priorité en utilisant la fonction `priority`. Le pixel avec la priorité maximale est sélectionné. 
    Arguments: 
    - front_pixels_mask (numpy.ndarray) : Masque binaire indiquant les pixels appartenant à la frontière de la région cible (1 pour les pixels de la frontière, 0 sinon). 
    - new_image (numpy.ndarray) : Image mise à jour utilisée pour calculer les gradients et la priorité. 
    - original_image (numpy.ndarray) : Image originale avant restauration, utilisée si nécessaire pour calculer les priorités. 
    - target_region_mask (numpy.ndarray) : Masque binaire indiquant la région cible à restaurer (1 pour les pixels à remplir, 0 sinon). 
    - confidence_matrix (numpy.ndarray) : Matrice de confiance des pixels, avec des valeurs entre 0 et 1. 
    - image_size (tuple ou list de deux entiers) : Dimensions de l'image (largeur, hauteur). 
    - patch_size (int) : Taille du patch carré centré autour de chaque pixel. 
    Valeurs de retour: tuple : 
    - pixel_max (list ou tuple) : Coordonnées (x, y) du pixel avec la priorité maximale. 
    - max_confidence (float) : Confiance associée au pixel sélectionné. 
    - max_data_term (float) : Terme de données associé au pixel sélectionné. 
    - max_priority (float) : Priorité maximale trouvée parmi les pixels de la frontière. 
    """
    max_confidence = 0.
    max_data_term = 0.
    max_priority = 0.

    front_pixels_list = list_front_pixels(front_pixels_mask) # Liste des pixels de la frontière créée avec la fonction précédemment définie
    pixel_max = front_pixels_list[0] # Initialisation arbitraire du pixel de priorité maximale
    for pixel in front_pixels_list: # Boucle sur tous les pixels de la frontière
        # On appelle priority pour récuperer la priorité du pixel courant
        pixel_confidence, pixel_data_term, pixel_priority = priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, new_image)
        
        if pixel_priority > max_priority: # On ccompare la priorité calculée précédemment avec la valeur maximum enregistrée jusqu'à maintenant
            # Si la priorité est supérieure, on change les valeurs des variables max_priority, pixel_max, max_confidence et max_data_term avec les informations du pixel courant
            max_priority = pixel_priority
            pixel_max = pixel
            max_confidence = pixel_confidence
            max_data_term = pixel_data_term

    return pixel_max, max_confidence, max_data_term, max_priority
    
def show_patchs_chosen(pixel, p_patch, q_patch):
    """ 
    Affiche les deux patchs impliqués dans une itération de l'algorithme d'inpainting. Cette fonction affiche côte à côte : 
    - Le patch à remplacer (centré sur le pixel spécifié). 
    - Le patch choisi pour remplir le patch à remplacer. 
    Arguments: 
    - pixel (tuple ou list de deux entiers) : Coordonnées (x, y) du pixel central du patch à remplacer. 
    - p_patch (numpy.ndarray) : Patch à remplacer (portion de l'image cible contenant des pixels à reconstruire). 
    - q_patch (numpy.ndarray) : Patch choisi (portion de l'image source utilisée pour combler les pixels manquants). 
    Valeur de retour: 
    - None : Affiche les patchs dans une fenêtre graphique. 
    """
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(p_patch, cmap='grey', vmin=0, vmax=255)
    axs[0].set_title(f"patch to replace (pixel = {pixel})")
    axs[1].imshow(q_patch, cmap='grey', vmin=0, vmax=255)
    axs[1].set_title("patch chosen")
    plt.show()