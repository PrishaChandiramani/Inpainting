import prisha_functions as pf
import luciano_functions as lf


import numpy as np
from PIL import Image

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

#test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 3)
#test4_image =Image.fromarray(test4)

#test4_image.show()

#test5 = pf.patch_search_compatible_niterations(target_region_mask2, gray_image_matrix,3,196)
#test5_image =Image.fromarray(test5)

#test5_image.show()

image2 = Image.open("./images/a-b.ppm.png")

image_matrix2 = np.array(image2)


gray_image2 = image2.convert("L")

gray_image_matrix2 = np.array(gray_image2)

print("gray image matrix 2 shape: ",gray_image_matrix2.shape)
target_region_mask3 = np.array([[False for i in range(gray_image_matrix2.shape[0])] for j in range(gray_image_matrix2.shape[1])])
target_region_mask3[52:72,52:72] = True 

image_initiale_matrix2 = gray_image_matrix2.copy()
image_initiale_matrix2[48:78,48:78] = 255
image_initiale2 = Image.fromarray(image_initiale_matrix2)
image_initiale2.show()

test6 = pf.patch_search_compatible(target_region_mask3, gray_image_matrix2, 3)
test6_image =Image.fromarray(test6)

test6_image.show()