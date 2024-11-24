import prisha_optimal_functions2 as pf
import prisha_optimal_functions_color as pc

import image_compression as ic
import color_compression as cc


import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import skimage.morphology as morpho
import matplotlib.pyplot as plt

import time

disk = morpho.disk(30)
print(disk.shape)


def test1():
    # test image triangle
    image = Image.open("./images_tests/image_triangle.png")
    image_matrix1 = np.array(image)
    image_matrix = image_matrix1[:,:,:3]

    #masque avec du bruit
    
    mask = Image.open("./images_tests/mask_triangle.png")
    mask_matrix1 = np.array(mask)
    mask_matrix=mask_matrix1[:,:,:3]
    mask_matrix_size = mask_matrix.shape
    target_region_mask = np.array([[[False] for i in range(mask_matrix_size[1])] for j in range(mask_matrix_size[0])])
    for i in range(mask_matrix_size[0]):
        for j in range(mask_matrix_size[1]):
            if np.all(mask_matrix[i,j] < np.array([30,30,30])):
                target_region_mask[i,j,0] = True

    
    print(" image matrix shape: ",image_matrix.shape)

    
    # mask rectangle

    #target_region_mask2 = np.array([[[False] for i in range(image_matrix.shape[1])] for j in range(image_matrix.shape[0])])
    #target_region_mask2[35:80,85:125,0] = True 

    #mask disque
    
    #target_region_mask = np.array([[[False] for i in range(image_matrix.shape[1])] for j in range(image_matrix.shape[0])])
    #target_region_mask[35:96,85:146,0] = disk


    image_initiale_matrix = image_matrix.copy()*target_region_mask
    
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()

    start_time = time.time()

    test4 = pc.patch_search_compatible(target_region_mask, image_matrix, 9)

    end_time = time.time()

    test4_image =Image.fromarray(test4)

    test4_image.show()

    print("Temps d'execution: {end_time - start_time:.4f} secs")

    return True

val = test1()