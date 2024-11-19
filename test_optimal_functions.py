import prisha_optimal_functions2 as pf
import prisha_optimal_functions_color as pc

import image_compression as ic
import color_compression as cc


import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from skimage.morphology import binary_dilation



def test1():
    # test trait hotizontal
    image = Image.open("./images/a-b-c.ppm.png")
    image_matrix = np.array(image)
    gray_image = image.convert("L")
    gray_image_matrix = np.array(gray_image)
    print("gray image matrix shape: ",gray_image_matrix.shape)

    target_region_mask2 = np.array([[False for i in range(gray_image_matrix.shape[1])] for j in range(gray_image_matrix.shape[0])])
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

    target_region_mask2 = np.array([[False for i in range(gray_image_matrix.shape[1])] for j in range(gray_image_matrix.shape[0])])
    target_region_mask2[100:250,100:220] = True 

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix[100:250,100:220] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 9)
    test4_image =Image.fromarray(test4)

    test4_image.show()

    return True

def test3():
    # test image paper
    gray_image_matrix1 = ic.get_compressed_image("./images/image1.jpg",2)
    gray_image_matrix = gray_image_matrix1[:,:]

    

    print("gray image matrix shape: ",gray_image_matrix.shape)
    target_region_mask2 = np.array([[False for i in range(gray_image_matrix.shape[1])] for j in range(gray_image_matrix.shape[0])])
    target_region_mask2[40:180,20:130] = True 
    print("target_region_mask.shape : ",target_region_mask2.shape)

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix[40:180,20:130] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 9)
    test4_image =Image.fromarray(test4)

    test4_image.show()

    return True

def test4():
    # test image paper
    image = Image.open("./images/image2.jpg")
    image_matrix = np.array(image)
    gray_image = image.convert("L")
    gray_image_matrix = np.array(gray_image)
    print("gray image matrix shape: ",gray_image_matrix.shape)

    target_region_mask2 = np.array([[False for i in range(gray_image_matrix.shape[1])] for j in range(gray_image_matrix.shape[0])])
    target_region_mask2[80:120,80:120] = True 

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix[80:120,80:120] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 3)
    test4_image =Image.fromarray(test4)

    test4_image.show()
    
    return True


def test5():
    image = Image.open("./images/image5.jpg")
    image_matrix = np.array(image)
    gray_image = image.convert("L")
    gray_image_matrix2 = np.array(gray_image)
    gray_image_matrix = gray_image_matrix2[6:,:]
    print("gray image matrix shape: ",gray_image_matrix.shape)

    mask = Image.open("./images/mask5.jpg")
    
    target_region = mask.convert("L")
    target_region_mask = np.array(target_region)
    target_region_mask2 = target_region_mask[6:,:]
    
    print("mask matrix shape: ",target_region_mask2.shape)

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix = image_initiale_matrix*target_region_mask2
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 9)
    test4_image =Image.fromarray(test4)

    test4_image.show()


def test6():
    image = Image.open("./images/image4.jpg")
    image_matrix = np.array(image)
    gray_image = image.convert("L")
    gray_image_matrix2 = np.array(gray_image)
    gray_image_matrix = gray_image_matrix2[:342,:]
    print("gray image matrix shape: ",gray_image_matrix.shape)

    mask = Image.open("./images/mask4.jpg")
    
    target_region = mask.convert("L")
    target_region_mask = np.array(target_region)
    target_region_mask2 = target_region_mask[:,:342]
    
    print("mask matrix shape: ",target_region_mask2.shape)

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix = image_initiale_matrix*target_region_mask2
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible(target_region_mask2, gray_image_matrix, 9)
    test4_image =Image.fromarray(test4)

    test4_image.show()


def test7():

    # test image paper
    image = Image.open("./images/image1.jpg")
    image_matrix1 = np.array(image)
    
    image_matrix2 = cc.compression(image_matrix1,2)
    image_matrix = gaussian_filter(image_matrix2,1)
    print(" image matrix shape: ",image_matrix.shape)
    
    image_matrix = np.clip(image_matrix,0,255).astype(np.uint8)
    compressed_image = Image.fromarray(image_matrix)
    compressed_image.show()
    print(" image matrix shape: ",image_matrix.shape)

    target_region_mask2 = np.array([[[False] for i in range(image_matrix.shape[1])] for j in range(image_matrix.shape[0])])
    target_region_mask2[100:237,15:130,0] = True 
    print("target_region_mask.shape : ",target_region_mask2.shape)

    image_initiale_matrix = image_matrix.copy()
    image_initiale_matrix[100:240,15:130] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pc.patch_search_compatible(target_region_mask2, image_matrix, 9)
    test4_image =Image.fromarray(test4)

    test4_image.show()

    return True

def test_calcul_dist_couleur():
    p = np.array([[[1,2,3] for i in range(9)]for j in range(9)])
    q = np.array([[[4,5,6] for i in range(9)] for j in range(9)])
    p_mask = np.array([[[True,True,True] ,[False,False,False],[True,True,True],[True,True,True],[False,False,False],[True,True,True],[True,True,True],[False,False,False],[False,False,False]] for i in range (9)])
    dist = pf.calcul_dist_couleur(p,q,p_mask)
    return dist

def test8():
    # test trait hotizontal
    image = Image.open("./images/a-b-c.ppm.png")
    image_matrix1 = np.array(image)
    image_matrix = image_matrix1[:,:,:3]
    print(" image matrix shape: ",image_matrix.shape)

    target_region_mask2 = np.array([[[False] for i in range(image_matrix.shape[1])] for j in range(image_matrix.shape[0])])
    target_region_mask2[16:44,16:44,0] = True 

    image_initiale_matrix = image_matrix.copy()
    image_initiale_matrix[16:44,16:44] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pc.patch_search_compatible(target_region_mask2, image_matrix, 3)
    test4_image =Image.fromarray(test4)

    test4_image.show()

    return True

def test_neighbours():
    image_matrix = np.array([[[3,1,2],[4,5,6],[7,8,9]],[[3,1,2],[4,5,6],[7,8,9]],[[3,1,2],[4,5,6],[7,8,9]],[[3,1,2],[4,5,6],[7,8,9]],[[3,1,2],[4,5,6],[7,8,9]],[[3,1,2],[4,5,6],[7,8,9]]])
    print(image_matrix.shape)
    target_region_mask = np.array([[[False] for i in range(15)] for j in range(15)])
    target_region_mask[6:9,6:9] = True
    print(target_region_mask.shape)
    neighbours = pc.neighbour_to_source_region(9,9,target_region_mask)
    n1 = pc.neighbour_to_source_region(7,6,target_region_mask)
    print(n1)
    print(neighbours)
    return True




#a = test_calcul_dist_couleur()
#print(a)
val = test7()

#mask = Image.open("./images/mask5.jpg")
#target_region = mask.convert("L")
#target_region_mask = np.array(target_region)
#target_region_mask2 = target_region_mask[6:,:]
#print(target_region_mask2)


"""
# Example matrix
matrix = np.array([
    [False,False, False, False, False,False],
    [False,False, True, True, False, False],
    [False,False, True, True, False,False],
    [False,False, False, False, False,False]
])

# Define the structuring element (3x3 kernel)
structuring_element = np.array([
    [False, True, True,False],
    [True, True, True,True],
    [True, True, True,True],
    [False, True, True,False]
])

# Apply binary dilation
dilated_matrix = binary_dilation(matrix, structuring_element)

print(dilated_matrix)
"""