import numpy as np
from PIL import Image

# Partie codée par Prisha


def compression(image_matrix,n):
    ns1= image_matrix.shape[0]//n
    ns2=image_matrix.shape[1]//n
    compressed_image = np.array([[[0,0,0] for i in range(ns2)] for j in range(ns1)])
    for i in range(0,image_matrix.shape[0]-1,n):
        for j in range(0,image_matrix.shape[1]-1,n):
            compressed_image[i//n-1,j//n-1] = image_matrix[i,j]
    return compressed_image