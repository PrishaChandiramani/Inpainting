import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal

# Partie codée par Luciano

def region3x3(x, y, img, target_region_mask, maxdepth):
    """ 
    Renvoie une région de 3x3 pixels entièrement située dans la région source et proche du pixel spécifié, en s'assurant que la zone ne contient pas de pixels de la région cible. La fonction ajuste dynamiquement la position du pixel central de la région 3x3 si elle contient des pixels de la région cible. Un paramètre de profondeur maximale (`maxdepth`) est utilisé pour éviter les boucles infinies. 
    Arguments: 
    - x (int) : Coordonnée x (ligne) du pixel central initial. 
    - y (int) : Coordonnée y (colonne) du pixel central initial. 
    - img (numpy.ndarray) : Image source à partir de laquelle la région sera extraite. 
    - target_region_mask (numpy.ndarray) : Masque binaire indiquant la région cible (1 pour les pixels à remplir, 0 pour les autres). 
    - maxdepth (int) : Nombre maximal d'itérations pour ajuster la position de la région 3x3. 
    Valeur de retour: 
    - numpy.ndarray : Région 3x3 extraite contenant uniquement des pixels de la région source. Si le pixel est proche des bords ou si la profondeur maximale est atteinte sans trouver une région source valide, une matrice 3x3 de zéros est renvoyée. 
    """
    size_x = img.shape[0]
    size_y = img.shape[1]
    
    if x == 0 or y == 0 or x == size_x - 1 or y == size_y - 1: # On écarte les pixels au bords pour éviter les erreurs
        return np.zeros((3, 3))

    # Calcul des coordonnées des quatres coins de la région 3x3 considérée
    xmin, xmax = x - 1, x + 1
    ymin, ymax = y - 1, y + 1

    # Matrices en x et en y qui permettent de calculer la direction du décalage à réaliser pour se retrouver dans la région source 
    x_matrix = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    y_matrix = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    local_target_region_mask = target_region_mask[xmin:xmax + 1, ymin:ymax + 1] # Masque local de la région cible autour du pixel considéré 

    depth = maxdepth
    while depth > 0 and np.any(local_target_region_mask): 
        # Boucle qui va se répéter jusqu'à ce que la région 3x3 considérée soit compris entièrement dans la région source,
        # ou que l'on ai dépassé le nombres maximum d'itérations
        
        if x == 1 or x == size_x - 2 or y == 1 or y == size_y - 2: # On écarte les cas où l'on est trop proche du bord, ce qui peux entrainer des erreurs
            break

        # On calcule le décalage nécessaire en x et en y pour s'éloigner de la région cible 
        x_offset = np.sign(np.sum(x_matrix * local_target_region_mask))
        y_offset = np.sign(np.sum(y_matrix * local_target_region_mask))
        # On met à jour les coordonnées du centre de la région considérée en fonction de ces décalages
        x += x_offset
        y += y_offset
        
        # Mise à jour des coordonées des coins de la région 3x3 considérée
        xmin, xmax = x - 1, x + 1
        ymin, ymax = y - 1, y + 1

        # Mise à jour du masque local de la région cible
        local_target_region_mask = target_region_mask[xmin:xmax + 1, ymin:ymax + 1]
        depth -= 1

    result = img[xmin:xmax + 1, ymin:ymax + 1] # Région 3x3 qui contient seulement des pixels de la région source (sauf dans le cas des bords)
    return result
    
def new_gradient(pixel, image, target_region_mask):
    """ 
    Calcule le gradient (en x et y) pour une région connue de l'image proche du pixel donné. Cette fonction utilise des matrices de Sobel pour calculer les gradients dans une région 3x3 proche du pixel spécifié. Si le pixel se trouve dans une région cible, une région "connue" proche est recherchée à l'aide de la fonction `region3x3`. 
    Arguments: 
    - pixel (tuple ou list de deux entiers) : Coordonnées (x, y) du pixel où le gradient doit être calculé. 
    - image (numpy.ndarray) : Image utilisée pour calculer le gradient. 
    - target_region_mask (numpy.ndarray) : Masque binaire indiquant la région cible (1 pour les pixels à remplir, 0 pour les autres). 
    Valeur de retour: 
    - list : Liste contenant les deux composantes du gradient [gradient_x, gradient_y]. 
    """
    gradient = [0. , 0.] # Initialisation du gradient
    x, y = pixel[0], pixel[1]

    pixel_region = region3x3(x, y, image, target_region_mask, 3) # Recherche de la région "connue" proche du pixel considéré et où l'on peut calculer le gradient 
    
    # Matrices de Sobel pour le calcul du gradient en x et en y :
    gradient_core_x = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) 
    gradient_core_y = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Calcul des valeurs de gradient en x et en y à l'aide des matrices de Sobel sur la région 3x3 trouvée précédemment
    gradient[0] = np.sum(gradient_core_x * pixel_region)
    gradient[1] = np.sum(gradient_core_y * pixel_region)

    return gradient

def new_orthogonal_front_vector(pixel, target_region_mask):
    """ 
    Calcule le vecteur normal à la frontière de la région cible au point spécifié par le pixel donné. Cette fonction utilise des matrices de Sobel pour calculer le gradient dans une région 3x3 autour du pixel, et en déduit un vecteur orthogonal à la frontière de la région cible. Le vecteur est ensuite normalisé. 
    Arguments: 
    - pixel (tuple ou list de deux entiers) : Coordonnées (x, y) du pixel où le vecteur normal doit être calculé. 
    - target_region_mask (numpy.ndarray) : Masque binaire indiquant la région cible (1 pour les pixels à remplir, 0 pour les autres). 
    Valeur de retour: 
    - list : Liste contenant les composantes du vecteur normal [vecteur_x, vecteur_y], normalisé. 
    """
    
    orthogonal_front_vector = [0. , 0.] # Initialisation du vecteur normal
    x, y = pixel[0], pixel[1]

    pixel_region = region3x3(x, y, target_region_mask, target_region_mask, 0) # Région 3x3 sur laquelle on va caluler le gradient (maxdepth = 0 car on ne veut pas se décaler)
    # Matrices de Sobel pour le calcul du gradient en x et en y :
    orthogonal_front_vector_core_x = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    orthogonal_front_vector_core_y = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # Calcul des valeurs de gradient en x et en y à l'aide des matrices de Sobel sur la région 3x3 trouvée précédemment
    orthogonal_front_vector[0] = np.sum(orthogonal_front_vector_core_x * pixel_region)
    orthogonal_front_vector[1] = np.sum(orthogonal_front_vector_core_y * pixel_region)
    # On normalise le vecteur obtenu en s'assurant de ne pas diviser par 0
    if np.sqrt(orthogonal_front_vector[0] ** 2 + orthogonal_front_vector[1] ** 2) > 0:
        orthogonal_front_vector = orthogonal_front_vector / np.sqrt(orthogonal_front_vector[0] ** 2 + orthogonal_front_vector[1] ** 2)
    return orthogonal_front_vector

    