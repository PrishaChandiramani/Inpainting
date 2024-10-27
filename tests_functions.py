import prisha_functions as pf
import luciano_functions as lf


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



def test1():
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

    test5 = pf.patch_search_compatible_niterations(target_region_mask2, gray_image_matrix,3,196)
    test5_image =Image.fromarray(test5)

    test5_image.show()
    
    return True

def test3():

    # problème lorsque deux patchs sont à la même distance alors que l'un est optimal et pas l'autre, si le non optimal est trouvé en premier, c'est lui qu'on utilisera : part d'aléatoire
    image2 = Image.open("./images/a-b.ppm.png")
    image_matrix2 = np.array(image2)
    gray_image2 = image2.convert("L")
    gray_image_matrix2 = np.array(gray_image2)

    print("gray image matrix 2 shape: ",gray_image_matrix2.shape)
    target_region_mask3 = np.array([[False for i in range(gray_image_matrix2.shape[0])] for j in range(gray_image_matrix2.shape[1])])
    target_region_mask3[48:78,48:78] = True 

    image_initiale_matrix2 = gray_image_matrix2.copy()
    image_initiale_matrix2[48:78,48:78] = 255
    image_initiale2 = Image.fromarray(image_initiale_matrix2)
    image_initiale2.show()

    test6 = pf.patch_search_compatible_niterations(target_region_mask3, gray_image_matrix2, 9, 250)
    test6_image =Image.fromarray(test6)

    test6_image.show()
    return True



def test4():

    image2 = Image.open("./images/a-b.ppm.png")
    image_matrix = np.array(image2)
    
    gray_image2 = image2.convert("L")
    gray_image_matrix = np.array(gray_image2)
    gray_image_matrix2 = gray_image_matrix[20:108,20:108]

    print("gray image matrix 2 shape: ",gray_image_matrix2.shape)
    target_region_mask3 = np.array([[False for i in range(gray_image_matrix2.shape[0])] for j in range(gray_image_matrix2.shape[1])])
    target_region_mask3[30:60,30:60] = True 

    image_initiale_matrix2 = gray_image_matrix2.copy()
    image_initiale_matrix2[30:60,30:60] = 255
    image_initiale2 = Image.fromarray(image_initiale_matrix2)
    image_initiale2.show()

    test6 = pf.patch_search_compatible_niterations(target_region_mask3, gray_image_matrix2, 3, 250)
    test6_image =Image.fromarray(test6)

    test6_image.show()
    return True

def test5(): 
    image2 = Image.open("./images/b-c.ppm.png")
    image_matrix2 = np.array(image2)
    gray_image2 = image2.convert("L")
    gray_image_matrix = np.array(gray_image2)
    gray_image_matrix2 = gray_image_matrix[:62,:]

    print("gray image matrix 2 shape: ",gray_image_matrix2.shape)
    target_region_mask3 = np.array([[False for i in range(gray_image_matrix2.shape[0])] for j in range(gray_image_matrix2.shape[1])])
    target_region_mask3[15:45,15:45] = True 

    image_initiale_matrix2 = gray_image_matrix2.copy()
    image_initiale_matrix2[15:45,15:45] = 0
    image_initiale2 = Image.fromarray(image_initiale_matrix2)
    image_initiale2.show()

    test6 = pf.patch_search_compatible_niterations(target_region_mask3, gray_image_matrix2, 3, 250)
    test6_image =Image.fromarray(test6)

    test6_image.show()
    return True

def test6():
    gray_image_matrix1 = get_compressed_image("./images/filled_with_mask.png",3)

    gray_image_matrix = gray_image_matrix1[:,76:423]

    print("gray image matrix shape: ",gray_image_matrix.shape)

    target_region_mask2 = np.array([[False for i in range(gray_image_matrix.shape[0])] for j in range(gray_image_matrix.shape[1])])
    target_region_mask2[100:250,100:220] = True 

    image_initiale_matrix = gray_image_matrix.copy()
    image_initiale_matrix[100:250,100:220] = 255
    image_initiale = Image.fromarray(image_initiale_matrix)
    image_initiale.show()
    test4 = pf.patch_search_compatible_niterations(target_region_mask2, gray_image_matrix, 15,500)
    test4_image =Image.fromarray(test4)

    test4_image.show()

    return True


#Val = test6()
#val = compression_test("./images/filled_with_mask.png",3) 
#im = get_compressed_image("./images/filled_with_mask.png",3)
#new_image = Image.fromarray(im[:,76:423])
#new_image.show()



# pour le calcul de gradient, tricher et faire sur la vraie image, ou calculer sur un grand patch et prendre la moyenne ou se décaler sur l'image
# fonctionne qu'avec des patchs et images carrés pour le moment