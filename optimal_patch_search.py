from PIL import Image
import numpy as np

image = Image.open("./images/a-b-c.ppm.png")

image_matrix = np.array(image)

#print(image_matrix.shape)

#print(image_matrix)

#image.show()

gray_image = image.convert("L")

gray_image_matrix = np.array(gray_image)

print("gray image matrix shape: ",gray_image_matrix.shape)

#print(gray_image_matrix)

#gray_image.show()

#p = gray_image_matrix[30:39,30:39]
#print(p.shape)
#print(p)
#p_image = Image.fromarray(p)
#p_image.show()

## Fonctions

def calcul_dist(p,q, p_mask):
   # dans p, il peut y avoir des valeurs à None
   #p_mask[i,j] = True si il faut prendre la valeur du pixel dans p qu'elle n'est pas vide
    sum = 0
    if p.shape != q.shape:
        raise ValueError("Les deux patchs n'ont pas la même taille")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p_mask[i,j] == True:
                 # Cast to float64 to prevent overflow
                diff = float(p[i, j]) - float(q[i, j])
                sum += diff ** 2
    return sum

def choose_q(target_region_mask, p,p_mask,im, patch_size):
    D={}
    #psi = gray_image_matrix - omega #déterminer comment déterminer psi à partir de omega
    #print(psi)
    #psi_image = Image.fromarray(psi)
    #psi_image.show()
    #print("psi.shape:",psi.shape)
    source_region_mask = np.logical_not(target_region_mask)
    print("source_region_mask.shape:",source_region_mask.shape)
    for i in range(source_region_mask.shape[0]-patch_size):
        for j in range(source_region_mask.shape[1]-patch_size):
            q_mask = source_region_mask[i:i+patch_size,j:j+patch_size]
            #print("q_mask.shape:",q_mask.shape)
            valid_patch = True
            for k in range(patch_size):
                for l in range(patch_size):
                    if q_mask[k,l] == False:
                        valid_patch = False #on peut mettre un break après pour minimiser le nombre d'itérations
            if valid_patch:
                q = im[i:i+patch_size,j:j+patch_size]
                d = calcul_dist(p,q,p_mask)
                D[(i,j)]=d
    minimum_D = min(D, key=D.get) # renvoie la clé de la valeur minimale
    q_opt = im[minimum_D[0]:minimum_D[0]+patch_size,minimum_D[1]:minimum_D[1]+patch_size]

    return q_opt 

## Test

target_region_mask = np.array([[False for i in range(61)] for j in range(61)])
target_region_mask[12:48,12:48] = True 
print("target_region_shape: ",target_region_mask.shape)



new_matrix = gray_image_matrix.copy()

new_matrix_mask = new_matrix.copy()
for i in range(61):
    for j in range(61):
        if target_region_mask[i,j] == True:
            new_matrix_mask[i,j] = 0

new_matrix_image = Image.fromarray(new_matrix_mask)
new_matrix_image.show()

## écrire la fonction copy image data


p1 = new_matrix[8:17,8:17] # patch 9
p1_mask = np.array([[True for i in range(9)] for j in range(9)])
p1_mask [12:17,12:17] = False

p1_image = Image.fromarray(p1)
#p1_image.show()


D1 = choose_q(target_region_mask, p1, p1_mask, new_matrix,9)

D1_image = Image.fromarray(D1)
D1_image.show()


new_matrix[8:17,8:17] = D1
target_region_mask[8:17,8:17] = False

p2 = new_matrix[44:53,44:53] # patch 9
p2_mask = np.array([[True for i in range(9)] for j in range(9)])
p2_mask [44:48,44:48] = False # attention dépend de target region mask or à chaque itération, target region mask change, pour l'instant on ne prend pas ça en compte, mais il le faut

p2_image = Image.fromarray(p2)
#p2_image.show()


D2 = choose_q(target_region_mask, p2,p2_mask, new_matrix,9)

D2_image = Image.fromarray(D2)
D2_image.show()

new_matrix[44:53,44:53] = D2
target_region_mask[44:53,44:53] = False


p3 = new_matrix[8:17,44:53]
p3_mask = np.array([[True for i in range(9)] for j in range(9)])
p3_mask [12:17,44:48] = False

D3 = choose_q(target_region_mask, p3,  p3_mask, new_matrix,9)

D3_image = Image.fromarray(D3)
D3_image.show()

new_matrix[8:17,44:53] = D3

target_region_mask[8:12,44:53] = False



p4 = new_matrix[30:39,5:14]
p4_mask = np.array([[True for i in range(9)] for j in range(9)])
p4_mask [30:39,12:14] = False
D4 = choose_q(target_region_mask, p4,  p4_mask, new_matrix,9)

D4_image = Image.fromarray(D4)
D4_image.show()

new_matrix[30:39,5:14] = D4

target_region_mask[8:12,44:53] = False


new_matrix_image = Image.fromarray(new_matrix)
new_matrix_image.show()


## nouvelles fonctions


def update_target_region_mask(target_region_mask, selected_pixel, patch_size,im):
    new_target_region_mask = np.copy(target_region_mask)
    half_patch_size = patch_size // 2
    for x in range(max(selected_pixel[0] - half_patch_size, 0), min(selected_pixel[0] + half_patch_size + 1, im.shape[0] - 1)):
        for y in range(max(selected_pixel[1] - half_patch_size, 0), min(selected_pixel[1] + half_patch_size + 1, im.shape[1] - 1)):
            if target_region_mask[x, y]:
                new_target_region_mask[x, y] = False
    return new_target_region_mask

def patch_search(target_region_mask, im, patch_size):
    # changer im en im_matrix pour être plus cohérent
    new_matrix = im.copy()
    half_patch_size = patch_size // 2
    for i in range(target_region_mask.shape[0]):
        for j in range(target_region_mask.shape[1]):
            if target_region_mask[i,j] == True:
                patch = im[max(i - half_patch_size, 0):min(i+ half_patch_size + 1, im.shape[0] - 1),max(j - half_patch_size, 0):min(j + half_patch_size + 1, im.shape[1] - 1)]
                patch_mask = np.array([[True for i in range(patch_size)] for j in range(patch_size)])
                # on met à False les valeurs de patch_mask qui correspondent à des valeurs de target_region_mask à True
                #for k in range(patch_size): # modiifier ce qu'il y a à l'intérieur des range
                    #for l in range(patch_size):
                        #if target_region_mask[i,j] == True:
                            #patch_mask[k,l] = False
                q_patch = choose_q(target_region_mask, patch,  patch_mask, new_matrix, patch_size)
                new_matrix [max(i - half_patch_size, 0):min(i+ half_patch_size + 1, im.shape[0] - 1),max(j - half_patch_size, 0):min(j + half_patch_size + 1, im.shape[1] - 1)] = q_patch
                update_target_region_mask(target_region_mask, (i,j), patch_size,im)
    return new_matrix

## nouveaux tests
target_region_mask2 = np.array([[False for i in range(61)] for j in range(61)])
target_region_mask2[12:48,12:48] = True 

image_initiale_matrix = gray_image_matrix.copy()
image_initiale_matrix[12:48,12:48] = 0
image_initiale = Image.fromarray(image_initiale_matrix)
image_initiale.show()

# si patch_size = 9, c'est plus long à éxecuter
#test2 = patch_search(target_region_mask2, gray_image_matrix,9)
#test2 = patch_search(target_region_mask2, gray_image_matrix,3)

test2_image =Image.fromarray(test2)

test2_image.show()
