import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal
import os

def priority(pixel, target_region_mask, confidence_matrix, patch_size, image_size):
    confidence = 0
    pixel_x, pixel_y = pixel[0], pixel[1]
    half_patch_size = patch_size // 2
    for x in range(max(pixel_x - half_patch_size, 0), min(pixel_x + half_patch_size + 1, image_size - 1)):
        for y in range(max(pixel_y - half_patch_size, 0), min(pixel_y + half_patch_size + 1, image_size - 1)):
            if not target_region_mask[x, y]:
                confidence += confidence_matrix[x, y]
    confidence /= patch_size*patch_size
    return confidence

#im = np.sqrt(np.random.rand(10, 10))

img = Image.open('./Inpainting/circle.png')
img_array = np.array(ImageOps.grayscale(img))
print(img_array.shape)
image_size = img.size[0]

#print(img_array)

"""
target_region_mask = np.array([[i <= j for i in range(image_size)] for j in range(image_size)])
confidence_matrix = 1. - np.copy(target_region_mask)
patch_size = 5

show_image(im, 'image originale')
#show_image(np.ma.masked_array(im, target_region_mask), 'image avec une partie enlevée')
#show_image(target_region_mask, 'région enlevée de l\'image originale')
#show_image(confidence_matrix, 'matrice de confiance des pixels')

def update_confidence(confidence_matrix, image_size, target_region_mask, patch_size):
    new_confidence_matrix = np.zeros((image_size, image_size))
    for x in range(image_size):
        for y in range(image_size):
            if confidence_matrix[x, y] == 0:
                new_confidence_matrix[x, y] = priority([x, y], target_region_mask, confidence_matrix, patch_size, image_size)
            else:
                new_confidence_matrix[x, y] = confidence_matrix[x, y]
    return new_confidence_matrix

#show_image(confidence_matrix, 'matrice de confiance initiale')
#confidence_matrix = update_confidence(confidence_matrix, image_size, target_region_mask, patch_size)
#show_image(confidence_matrix, 'matrice de confiance après une étape')
"""
def show_image(im, title): # pour afficher une image
    plt.imshow(im, cmap='grey')
    plt.title(title)
    plt.show()

def compute_gradient(img):
    # calcule le gradient d'une image en niveau de gris
    gradient_matrix = np.zeros((img.shape[0], img.shape[1], 2))
    gradient_core_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_core_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_matrix[:, :, 0] = signal.convolve2d(img, gradient_core_x, mode='same', boundary='wrap')
    gradient_matrix[:, :, 1] = signal.convolve2d(img, gradient_core_y, mode='same', boundary='wrap')
    
    return np.abs(gradient_matrix)

gradient = compute_gradient(img_array)

def show_gradient(img):
    # calcule et affiche les vecteurs gradients d'une image en niveau de gris
    grad = compute_gradient(img)
    gradx, grady = grad[:, :, 0], grad[:, :, 1]
    X = np.arange(img.shape[0])
    Y = np.arange(img.shape[1])
    X, Y = np.meshgrid(X, Y)

    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, gradx, grady)

    plt.show()

#show_image(gradient[:, :, 0], "gradient en x")
#show_image(gradient[:, :, 1], "gradient en y")
show_gradient(img_array)