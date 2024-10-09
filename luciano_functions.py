import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal



def priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, gradient_matrix, orthogonal_vectors_matrix):
    
    # Calcul du terme de confiance
    confidence = 0
    pixel_x, pixel_y = pixel[0], pixel[1]
    half_patch_size = patch_size // 2
    for x in range(max(pixel_x - half_patch_size, 0), min(pixel_x + half_patch_size + 1, image_size - 1)):
        for y in range(max(pixel_y - half_patch_size, 0), min(pixel_y + half_patch_size + 1, image_size - 1)):
            if not target_region_mask[x, y]:
                confidence += confidence_matrix[x, y]
    confidence /= patch_size*patch_size

    # Calcul du terme de données
    data_term = np.abs(gradient_matrix[x, y, 0] * orthogonal_vectors_matrix[x, y, 0] + gradient_matrix[x, y, 1] * orthogonal_vectors_matrix[x, y, 1])
    data_term /= 255

    return confidence, confidence*data_term


def update_confidence(confidence_matrix, target_region_mask, selected_pixel, patch_size):
    new_confidence_matrix = np.copy(confidence_matrix)
    selected_pixel_confidence = confidence_matrix[selected_pixel[0], selected_pixel[1]]
    half_patch_size = patch_size // 2
    for x in range(max(selected_pixel[0] - half_patch_size, 0), min(selected_pixel[0] + half_patch_size + 1, image_size - 1)):
        for y in range(max(selected_pixel[1] - half_patch_size, 0), min(selected_pixel[1] + half_patch_size + 1, image_size - 1)):
            if target_region_mask[x, y]:
                new_confidence_matrix[x, y] = selected_pixel_confidence
    return new_confidence_matrix

def update_target_region_mask(target_region_mask, selected_pixel, patch_size):
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

def pixel_with_min_priority(front_pixels_mask, image, target_region_mask, confidence_matrix, image_size, patch_size):
    orthogonal_vectors_matrix = front_orthogonal_vectors(target_region_mask)
    gradient_matrix = compute_gradient(image * (1. - target_region_mask))
    
    pixel_min_confidence = 0.
    min_priority = 1.

    front_pixels_list = []

    for i in range(front_pixels_mask.shape[0]):
        for j in range(front_pixels_mask.shape[1]):
            if front_pixels_mask[i, j]:
                front_pixels_list.append([i, j])
    
    pixel_min = front_pixels_list[0]
    for pixel in front_pixels_list:
        pixel_confidence, pixel_priority = priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size, gradient_matrix, orthogonal_vectors_matrix)
        if pixel_priority < min_priority:
            pixel_min = pixel
            pixel_min_confidence = pixel_confidence

    return pixel_min, pixel_min_confidence

if __name__ == "__main__":

    img = Image.open('./Inpainting/circle.png')
    img_array = np.array(ImageOps.grayscale(img))
    image_size = img.size[0]

    target_region_mask = np.array([[i <= j for i in range(image_size)] for j in range(image_size)])
    confidence_matrix = 1. - np.copy(target_region_mask)
    patch_size = 5

    show_image(img_array, 'image originale')
    #show_image(np.ma.masked_array(im, target_region_mask), 'image avec une partie enlevée')
    show_image(target_region_mask, 'région enlevée de l\'image originale')
    #show_image(confidence_matrix, 'matrice de confiance des pixels')

    #show_image(confidence_matrix, 'matrice de confiance initiale')
    #confidence_matrix = update_confidence(confidence_matrix, image_size, target_region_mask, patch_size)
    #show_image(confidence_matrix, 'matrice de confiance après une étape')
    
    
    mask_gradient = compute_gradient(target_region_mask, boundary_mode='symm')
    #gradient = compute_gradient(img_array)
    #show_image(gradient[:, :, 0], "gradient en x")
    #show_image(gradient[:, :, 1], "gradient en y")
    
    #show_image(mask_gradient[:, :, 0], "gradient en x")
    #show_image(mask_gradient[:, :, 1], "gradient en y")
    #show_gradient_vectors(mask_gradient)
    show_one_gradient_vector([25, 25], mask_gradient[25, 25])

    #show_gradient_vectors(gradient)
