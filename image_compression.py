import numpy as np 
from PIL import Image

def compression(image_matrix,n):
    ns1= image_matrix.shape[0]//n
    ns2=image_matrix.shape[1]//n
    compressed_image = np.zeros((ns1,ns2))
    for i in range(0,image_matrix.shape[0]-1,n):
        for j in range(0,image_matrix.shape[1]-1,n):
            compressed_image[i//n-1,j//n-1] = image_matrix[i,j]
    return compressed_image

def get_compressed_image(path,n):
    image = Image.open(path)
    gray_image = image.convert("L")
    gray_image_matrix = np.array(gray_image)

    compressed_image = compression(gray_image_matrix,n)
    return compressed_image

def display_image(image_matrix):
    print("new image size :", image_matrix.shape)
    new_image = Image.fromarray(image_matrix)
    new_image.show()
    return ("displayed")

def compression_test(path,n):
    final_image_matrix = get_compressed_image(path,n)
    val = display_image(final_image_matrix)
    return "done"
