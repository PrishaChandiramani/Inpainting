import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import signal

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
im = image.imread('circle.png')
im = im[:, :, 0]
image_size = im.shape[0]

def show_image(im, title): # pour afficher une image
    plt.imshow(im, cmap='grey')
    plt.title(title)
    plt.show()

'''
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
'''

def compute_gradient(img):
    gradient_matrix = np.zeros((img.shape(0), img.shape(1), 2))
    gradient_core_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_core_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_matrix[:, :, 0] = signal.convolve2d(img, gradient_core_x, mode='same', boundary='fill')
    gradient_matrix[:, :, 1] = signal.convolve2d(img, gradient_core_y, mode='same', boundary='fill')
    
    return gradient_matrix

gradient = gradient(im)

show_image(gradient[0])
show_image(gradient[1])