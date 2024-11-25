import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal
from new_gradient import new_gradient, new_orthogonal_front_vector


def priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, new_image, front_orthogonal_vectors, orthogonal_to_gradient_matrix):
    # Calcul du terme de confiance
    confidence = 0
    pixel_x, pixel_y = pixel[0], pixel[1]
    half_patch_size = patch_size // 2
    xmin, xmax = max(pixel_x - half_patch_size, 0), min(pixel_x + half_patch_size + 1, image_size[0] - 1)
    ymin, ymax = max(pixel_y - half_patch_size, 0), min(pixel_y + half_patch_size + 1, image_size[1] - 1)
    
    mat = confidence_matrix[xmin:xmax, ymin:ymax]
    confidence = np.sum(mat) / ((xmax - xmin + 1) * (ymax - ymin + 1))

    
    # Calcul du terme de données
    
    gradient = new_gradient(pixel, new_image, target_region_mask)
    orthogonal_to_gradient = [- gradient[1], gradient[0]]
    front_orthogonal_vector = new_orthogonal_front_vector(pixel, target_region_mask)
    data_term = np.abs(orthogonal_to_gradient[0] * front_orthogonal_vector[0] + orthogonal_to_gradient[1] * front_orthogonal_vector[1])
    #print("data term : ", data_term, " gradient : ", gradient, " orthogonal vector : ", front_orthogonal_vector)

    #data_term = np.abs(orthogonal_to_gradient_matrix[pixel_x, pixel_y, 0] * front_orthogonal_vectors[pixel_x, pixel_y, 0] + orthogonal_to_gradient_matrix[pixel_x, pixel_y, 1] * front_orthogonal_vectors[pixel_x, pixel_y, 1])
    
    
    data_term /= 255

    return confidence, data_term, confidence*data_term


def update_confidence(confidence_matrix, target_region_mask, selected_pixel, selected_pixel_confidence, patch_size, image_size):
    new_confidence_matrix = np.copy(confidence_matrix)
    half_patch_size = patch_size // 2
    xmin = max(selected_pixel[0] - half_patch_size, 0)
    xmax = min(selected_pixel[0] + half_patch_size + 1, image_size[0] - 1)
    ymin = max(selected_pixel[1] - half_patch_size, 0)
    ymax = min(selected_pixel[1] + half_patch_size + 1, image_size[1] - 1)
    confidence_patch = (1 - target_region_mask[xmin:xmax, ymin:ymax]) * confidence_matrix[xmin:xmax, ymin:ymax]
    new_confidence_patch = target_region_mask[xmin:xmax, ymin:ymax] * selected_pixel_confidence + confidence_patch
    new_confidence_matrix[xmin:xmax, ymin:ymax] = new_confidence_patch
    show_patchs_chosen(selected_pixel, confidence_patch, new_confidence_patch)
    return new_confidence_matrix

def update_target_region_mask(target_region_mask, selected_pixel, patch_size,im):
    #print("in update_target_region_mask")
    updated_matrix = target_region_mask.copy()
    half_patch_size = patch_size // 2
    #updated_matrix[selected_pixel[0],selected_pixel[1]] = False
    updated_matrix [max(selected_pixel[0] - half_patch_size, 0): min(selected_pixel[0] + half_patch_size + 1, im.shape[0] - 1),max(selected_pixel[1] - half_patch_size, 0): min(selected_pixel[1] + half_patch_size + 1 , im.shape[1] - 1)]= np.array([[False for i in range (patch_size)] for j in range(patch_size)])
    return updated_matrix

def compute_gradient(img, boundary_mode='wrap'):
    # calcule le gradient d'une image en niveau de gris
    gradient_matrix = np.zeros((img.shape[0], img.shape[1], 2))
    gradient_core_x = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_core_y = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_matrix[:, :, 0] = signal.convolve2d(img, gradient_core_x, mode='same', boundary=boundary_mode)
    gradient_matrix[:, :, 1] = signal.convolve2d(img, gradient_core_y, mode='same', boundary=boundary_mode)
    
    return gradient_matrix


def front_orthogonal_vectors(target_region_mask):
    front_orthogonal_vectors = np.zeros((target_region_mask.shape[0], target_region_mask.shape[1], 2))
    mask_gradient = compute_gradient(target_region_mask)
    front_orthogonal_vectors = mask_gradient / np.max(np.abs(mask_gradient))
    return front_orthogonal_vectors

def list_front_pixels(front_pixels_mask):
    non_null_indices = np.nonzero(front_pixels_mask)
    non_null_indices_list = [[non_null_indices[0][i], non_null_indices[1][i]] for i in range(len(non_null_indices[0]))]
    return non_null_indices_list


def pixel_with_max_priority(front_pixels_mask, new_image, original_image, target_region_mask, confidence_matrix, image_size, patch_size):
    
    
    #orthogonal_vectors_matrix = front_orthogonal_vectors(target_region_mask)
    #gradient_matrix = compute_gradient(new_image * (1. - target_region_mask) + original_image * target_region_mask)
    orthogonal_to_gradient_matrix = np.zeros((image_size[0], image_size[1], 2))
    orthogonal_vectors_matrix = np.copy(orthogonal_to_gradient_matrix)
    #orthogonal_to_gradient_matrix[:, :, 0] = - gradient_matrix[:, :, 1]
    #orthogonal_to_gradient_matrix[:, :, 1] = gradient_matrix[:, :, 0]
    
    max_confidence = 0.
    max_data_term = 0.
    max_priority = 0.

    front_pixels_list = list_front_pixels(front_pixels_mask)
    #print(f"front pixels list : {front_pixels_list}")
    pixel_max = front_pixels_list[0]
    for pixel in front_pixels_list:
        pixel_confidence, pixel_data_term, pixel_priority = priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, new_image, orthogonal_vectors_matrix, orthogonal_to_gradient_matrix)
        #print(f"-- pixel : {pixel} | confidence : {pixel_confidence} | data term : {pixel_data_term} | priority : {pixel_priority}")
        if pixel_priority > max_priority:
            max_priority = pixel_priority
            pixel_max = pixel
            max_confidence = pixel_confidence
            max_data_term = pixel_data_term

    return pixel_max, max_confidence, max_data_term, max_priority



def neighbour_to_source_region(x, y, target_region_mask):
    source_region_mask = 1. - target_region_mask
    number_of_source_region_neighours = 0
    if x > 0 and x < source_region_mask.shape[0] - 1:
        if y > 0 and y < source_region_mask.shape[1] - 1:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y - 1]
    elif x == 0:
        if y > 0 and y < source_region_mask.shape[1] - 1:
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y - 1]
    else:
        if y > 0 and y < source_region_mask.shape[1] - 1:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y - 1]
    return number_of_source_region_neighours > 0

def front_detection(im, target_region_mask):
    if target_region_mask.shape != im.shape:
        raise ValueError('target_region_mask and im must have the same shape')
    if np.all(target_region_mask == np.array([[False for i in range(im.shape[0])] for j in range(im.shape[1])])):
        return ("No target region")
    else : 
        front = np.array([[False for i in range(im.shape[0])] for j in range(im.shape[1])])
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if target_region_mask[x, y]:
                    front[x, y] = neighbour_to_source_region(x, y, target_region_mask)
        return front
    
def show_patchs_chosen(pixel, p_patch, q_patch):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(p_patch, cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title(f"ancien patch de confiance (pixel = {pixel})")
    axs[1].imshow(q_patch, cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title("patch de confiance mis à jour")
    fig.colorbar(plt.cm.ScalarMappable(), ax=axs)
    plt.show()