import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal



def priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, gradient_matrix, orthogonal_vectors_matrix):
    
    # Calcul du terme de confiance
    confidence = 0
    pixel_x, pixel_y = pixel[0], pixel[1]
    half_patch_size = patch_size // 2
    for x in range(max(pixel_x - half_patch_size, 0), min(pixel_x + half_patch_size + 1, image_size[0] - 1)):
        for y in range(max(pixel_y - half_patch_size, 0), min(pixel_y + half_patch_size + 1, image_size[1] - 1)):
            confidence += confidence_matrix[x, y]
    confidence /= patch_size*patch_size
    confidence *= 10

    # Calcul du terme de données
    data_term = np.abs(gradient_matrix[x, y, 0] * orthogonal_vectors_matrix[x, y, 0] + gradient_matrix[x, y, 1] * orthogonal_vectors_matrix[x, y, 1])
    data_term /= 255
    data_term *= 100

    return confidence, data_term, confidence*data_term


def update_confidence(confidence_matrix, target_region_mask, selected_pixel, selected_pixel_confidence, patch_size, image_size):
    new_confidence_matrix = np.copy(confidence_matrix)
    half_patch_size = patch_size // 2
    for x in range(max(selected_pixel[0] - half_patch_size, 0), min(selected_pixel[0] + half_patch_size + 1, image_size[0] - 1)):
        for y in range(max(selected_pixel[1] - half_patch_size, 0), min(selected_pixel[1] + half_patch_size + 1, image_size[1] - 1)):
            if target_region_mask[x, y]:
                new_confidence_matrix[x, y] = selected_pixel_confidence
    return new_confidence_matrix

def update_target_region_mask(target_region_mask, selected_pixel, image_size, patch_size):
    new_target_region_mask = np.copy(target_region_mask)
    half_patch_size = patch_size // 2
    for x in range(max(selected_pixel[0] - half_patch_size, 0), min(selected_pixel[0] + half_patch_size + 1, image_size - 1)):
        for y in range(max(selected_pixel[1] - half_patch_size, 0), min(selected_pixel[1] + half_patch_size + 1, image_size - 1)):
            if target_region_mask[x, y]:
                new_target_region_mask[x, y] = False
    return new_target_region_mask


def show_image(im, title): # pour afficher une image
    plt.imshow(im, cmap='grey')
    plt.title(title)
    plt.show()

def compute_gradient(img, boundary_mode='wrap'):
    # calcule le gradient d'une image en niveau de gris
    gradient_matrix = np.zeros((img.shape[0], img.shape[1], 2))
    gradient_core_x = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_core_y = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_matrix[:, :, 0] = signal.convolve2d(img, gradient_core_x, mode='same', boundary=boundary_mode)
    gradient_matrix[:, :, 1] = signal.convolve2d(img, gradient_core_y, mode='same', boundary=boundary_mode)
    
    return gradient_matrix

def show_gradient_vectors(grad_matrix):
    # Affiche les vecteurs gradients d'une image en niveau de gris
    gradx, grady = grad_matrix[:, :, 0], - grad_matrix[:, :, 1]
    X = np.arange(grad_matrix.shape[1])
    Y = np.arange(grad_matrix.shape[0])
    X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    q = ax.quiver(X, Y, gradx, grady)
    ax.invert_yaxis()
    plt.show()

def show_one_gradient_vector(pixel, gradient):
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    q = ax.quiver(pixel[0], pixel[1], gradient[0], - gradient[1])
    ax.invert_yaxis()
    plt.show()

def front_orthogonal_vectors(target_region_mask):
    front_orthogonal_vectors = np.zeros((target_region_mask.shape[0], target_region_mask.shape[1], 2))
    mask_gradient = compute_gradient(target_region_mask)
    front_orthogonal_vectors = mask_gradient / np.max(np.abs(mask_gradient))
    return front_orthogonal_vectors

def pixel_with_max_priority(front_pixels_mask, image, target_region_mask, confidence_matrix, image_size, patch_size):
    orthogonal_vectors_matrix = front_orthogonal_vectors(target_region_mask)
    gradient_matrix = compute_gradient(image * (1. - target_region_mask) + 255 * target_region_mask)
    orthogonal_to_gradient_matrix = np.zeros((image_size[0], image_size[1], 2))
    orthogonal_to_gradient_matrix[:, :, 0] = - gradient_matrix[:, :, 1]
    orthogonal_to_gradient_matrix[:, :, 1] = gradient_matrix[:, :, 0]
    max_confidence = 0.
    max_data_term = 0.
    max_priority = 0.

    front_pixels_list = list_front_pixels(front_pixels_mask)
    pixel_max = front_pixels_list[0]
    for pixel in front_pixels_list:
        pixel_confidence, pixel_data_term, pixel_priority = priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, orthogonal_to_gradient_matrix, orthogonal_vectors_matrix)
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

def list_front_pixels(front_pixels_mask):
    front_pixels_list = []
    for x in range(front_pixels_mask.shape[0]):
        for y in range(front_pixels_mask.shape[1]):
            if front_pixels_mask[x, y]:
                front_pixels_list.append([x, y])
    return front_pixels_list

if __name__ == "__main__":

    img = Image.open('./Inpainting/circle.png')
    img_array = np.array(ImageOps.grayscale(img))
    image_size = img.size

    target_region_mask = np.array([[i <= j for i in range(image_size[0])] for j in range(image_size[1])])
    confidence_matrix = 1. - np.copy(target_region_mask)
    patch_size = 5
    front_mask = front_detection(img_array, target_region_mask)

    show_image(img_array, 'image originale')
    #show_image(np.ma.masked_array(im, target_region_mask), 'image avec une partie enlevée')
    show_image(target_region_mask, 'région enlevée de l\'image originale')
    #show_image(confidence_matrix, 'matrice de confiance des pixels')
    show_image(front_mask, 'contour de la target region')
    #show_image(confidence_matrix, 'matrice de confiance initiale')
    #confidence_matrix = update_confidence(confidence_matrix, image_size, target_region_mask, patch_size)
    #show_image(confidence_matrix, 'matrice de confiance après une étape')
    pixel_max, confidence, data_term, pixel_priority = pixel_with_max_priority(front_mask, img_array, target_region_mask, confidence_matrix, image_size[0], patch_size)
    print(f"pixel max trouvé : {pixel_max} | confiance : {confidence} | priorité : {pixel_priority}")
    show_image(confidence_matrix, "matrice de confiance avant")
    confidence_matrix = update_confidence(confidence_matrix, target_region_mask, pixel_max, confidence, patch_size, image_size)
    show_image(confidence_matrix, "matrice de confiance après")
    #print(list_front_pixels(front_mask))
    #mask_gradient = compute_gradient(target_region_mask, boundary_mode='symm')
    #gradient = compute_gradient(img_array)
    #show_image(gradient[:, :, 0], "gradient en x")
    #show_image(gradient[:, :, 1], "gradient en y")
    
    #show_image(mask_gradient[:, :, 0], "gradient en x")
    #show_image(mask_gradient[:, :, 1], "gradient en y")
    #show_gradient_vectors(mask_gradient)
    #show_one_gradient_vector([25, 25], mask_gradient[25, 25])

    #show_gradient_vectors(gradient)
