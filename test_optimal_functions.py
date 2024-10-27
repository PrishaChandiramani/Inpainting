import prisha_optimal_functions as pf

import image_compression as ic


import numpy as np
from PIL import Image



def test1():
    # test trait hotizontal
    image = Image.open("./images/a-b-c.ppm.png")
    image_matrix = np.array(image)
    gray_image = image.convert("L")
    gray_image_matrix = np.array(gray_image)
    print("gray image matrix shape: ",gray_image_matrix.shape)

    target_region_mask2 = np.array([[False for i in range(gray_image_matrix.shape[0])] for j in range(gray_image_matrix.shape[1])])
    target_region_mask2[16:44,16:44] = True 

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix[16:44,16:44] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 3)
    test4_image =Image.fromarray(test4)

    test4_image.show()

    return True





def test2():
    #test avec nounours
    gray_image_matrix1 = ic.get_compressed_image("./images/filled_with_mask.png",3)

    gray_image_matrix = gray_image_matrix1[:,76:423]

    print("gray image matrix shape: ",gray_image_matrix.shape)

    target_region_mask2 = np.array([[False for i in range(gray_image_matrix.shape[0])] for j in range(gray_image_matrix.shape[1])])
    target_region_mask2[100:250,100:220] = True 

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix[100:250,100:220] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 15)
    test4_image =Image.fromarray(test4)

    test4_image.show()

    return True

val = test2()