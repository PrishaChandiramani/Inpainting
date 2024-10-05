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

def choose_q(target_region_mask, p,p_mask,im):
    D={}
    #psi = gray_image_matrix - omega #déterminer comment déterminer psi à partir de omega
    #print(psi)
    #psi_image = Image.fromarray(psi)
    #psi_image.show()
    #print("psi.shape:",psi.shape)
    source_region_mask = np.logical_not(target_region_mask)
    print("source_region_mask.shape:",source_region_mask.shape)
    for i in range(source_region_mask.shape[0]-9):
        for j in range(source_region_mask.shape[1]-9):
            q_mask = source_region_mask[i:i+9,j:j+9]
            #print("q_mask.shape:",q_mask.shape)
            valid_patch = True
            for k in range(9):
                for l in range(9):
                    if q_mask[k,l] == False:
                        valid_patch = False #on peut mettre un break après pour minimiser le nombre d'itérations
            if valid_patch:
                q = im[i:i+9,j:j+9]
                d = calcul_dist(p,q,p_mask)
                D[(i,j)]=d
    minimum_D = min(D, key=D.get) # renvoie la clé de la valeur minimale
    q_opt = im[minimum_D[0]:minimum_D[0]+9,minimum_D[1]:minimum_D[1]+9]

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


D1 = choose_q(target_region_mask, p1, p1_mask, new_matrix)

D1_image = Image.fromarray(D1)
D1_image.show()


new_matrix[8:17,8:17] = D1
target_region_mask[8:17,8:17] = False

p2 = new_matrix[44:53,44:53] # patch 9
p2_mask = np.array([[True for i in range(9)] for j in range(9)])
p2_mask [44:48,44:48] = False # attention dépend de target region mask or à chaque itération, target region mask change, pour l'instant on ne prend pas ça en compte, mais il le faut

p2_image = Image.fromarray(p2)
#p2_image.show()


D2 = choose_q(target_region_mask, p2,p2_mask, new_matrix)

D2_image = Image.fromarray(D2)
D2_image.show()

new_matrix[44:53,44:53] = D2
target_region_mask[44:53,44:53] = False


p3 = new_matrix[8:17,44:53]
p3_mask = np.array([[True for i in range(9)] for j in range(9)])
p3_mask [12:17,44:48] = False

D3 = choose_q(target_region_mask, p3,  p3_mask, new_matrix)

D3_image = Image.fromarray(D3)
D3_image.show()

new_matrix[8:17,44:53] = D3

target_region_mask[8:12,44:53] = False



p4 = new_matrix[30:39,5:14]
p4_mask = np.array([[True for i in range(9)] for j in range(9)])
p4_mask [30:39,12:14] = False
D4 = choose_q(target_region_mask, p4,  p4_mask, new_matrix)

D4_image = Image.fromarray(D4)
D4_image.show()

new_matrix[30:39,5:14] = D4

target_region_mask[8:12,44:53] = False


new_matrix_image = Image.fromarray(new_matrix)
new_matrix_image.show()