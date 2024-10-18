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

test5 = pf.patch_search_compatible_niterations(target_region_mask2, gray_image_matrix,3,180)
test5_image =Image.fromarray(test5)

test5_image.show()

