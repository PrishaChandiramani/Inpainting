import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal
from new_gradient_final import new_gradient, new_orthogonal_front_vector


def priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, new_image):
    # Calcule la priorité du pixel passé en argument 
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
    # Renvoie la matrice de confiance mise à jour avec les valeurs de confiance des pixels reconstruits du patch autour de selected_pixel
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
    # Calcule les gradients en x et en y de tous les pixels de l'image passée en argument, et renvoie le résultat sous forme de numpy array de dimensions (image_size_x, image_size_y, 2)
    gradient_matrix = np.zeros((img.shape[0], img.shape[1], 2)) # On crée le array qui enregistrera les résultats
    # On définit les noyaux de Sobel en x et en y :
    gradient_core_x = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_core_y = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # On utilise la fonction convolve2d de Numpy pour calculer la convolution
    gradient_matrix[:, :, 0] = signal.convolve2d(img, gradient_core_x, mode='same', boundary=boundary_mode)
    gradient_matrix[:, :, 1] = signal.convolve2d(img, gradient_core_y, mode='same', boundary=boundary_mode)
    
    return gradient_matrix

def front_orthogonal_vectors(target_region_mask):
    # Calcule les gradients en x et en y de tous les pixels en considérant comme image le masque de la région cible, et renvoie le résultat sous forme de numpy array de dimensions (image_size_x, image_size_y, 2)
    front_orthogonal_vectors = np.zeros((target_region_mask.shape[0], target_region_mask.shape[1], 2)) # On crée le array qui enregistrera les résultats
    mask_gradient = compute_gradient(target_region_mask) # On appelle la fonction précédente compute_gradient()
    front_orthogonal_vectors = mask_gradient / np.abs(mask_gradient) # On normalise le résultat
    return front_orthogonal_vectors

def list_front_pixels(front_pixels_mask):
    non_null_indices = np.nonzero(front_pixels_mask)
    non_null_indices_list = [[non_null_indices[0][i], non_null_indices[1][i]] for i in range(len(non_null_indices[0]))]
    return non_null_indices_list


def pixel_with_max_priority(front_pixels_mask, new_image, original_image, target_region_mask, confidence_matrix, image_size, patch_size):
    
    
    max_confidence = 0.
    max_data_term = 0.
    max_priority = 0.

    front_pixels_list = list_front_pixels(front_pixels_mask)
    #print(f"front pixels list : {front_pixels_list}")
    pixel_max = front_pixels_list[0]
    for pixel in front_pixels_list:
        pixel_confidence, pixel_data_term, pixel_priority = priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, new_image)
        #print(f"-- pixel : {pixel} | confidence : {pixel_confidence} | data term : {pixel_data_term} | priority : {pixel_priority}")
        if pixel_priority > max_priority:
            max_priority = pixel_priority
            pixel_max = pixel
            max_confidence = pixel_confidence
            max_data_term = pixel_data_term

    return pixel_max, max_confidence, max_data_term, max_priority
    
def show_patchs_chosen(pixel, p_patch, q_patch):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(p_patch, cmap='grey', vmin=0, vmax=255)
    axs[0].set_title(f"patch to replace (pixel = {pixel})")
    axs[1].imshow(q_patch, cmap='grey', vmin=0, vmax=255)
    axs[1].set_title("patch chosen")
    plt.show()