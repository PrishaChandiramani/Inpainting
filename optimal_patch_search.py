from PIL import Image
import numpy as np

image = Image.open("./images/a-b-c.ppm.png")

image_matrix = np.array(image)

#print(image_matrix.shape)

#print(image_matrix)

#image.show()

gray_image = image.convert("L")

gray_image_matrix = np.array(gray_image)

print(gray_image_matrix.shape)

print(gray_image_matrix)

gray_image.show()

p = gray_image_matrix[30:39,30:39]
print(p.shape)
print(p)
p_image = Image.fromarray(p)
p_image.show()

def calcul_dist(p,q):
   # dans p, il peut y avoir des valeurs à None
    sum = 0
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p[i,j] != None:
             sum += (p[i,j]-q[i,j])**2   
    return sum

def choose_q(target_region_mask, p, im):
    D={}
    #psi = gray_image_matrix - omega #déterminer comment déterminer psi à partir de omega
    #print(psi)
    #psi_image = Image.fromarray(psi)
    #psi_image.show()
    #print("psi.shape:",psi.shape)
    source_region_mask = np.logical_not(target_region_mask)
    for i in range(source_region_mask.shape[0]-9):
        for j in range(source_region_mask.shape[1]-9):
            q_mask = source_region_mask[i:i+9,j:j+9]
            if q_mask == np.array([[True for i in range(9)] for j in range(9)]):
                q = im[i:i+9,j:j+9]
                d = calcul_dist(p,q)
                D[(i,j)]=d
            #print("q.shape:",q.shape)
            #print("q:",q)
            #print("d:",d)
    minimum_D = min(D, key=D.get) # renvoie la clé de la valeur minimale
    q_opt = im[minimum_D[0]:minimum_D[0]+9,minimum_D[1]:minimum_D[1]+9]

    return q_opt 


#omega = np.zeros((61,61))
#omega[12:48,12:48] = gray_image_matrix[12:48,12:48] # patch 10

target_region_mask = np.array([[False for i in range(61)] for j in range(61)])
target_region_mask[12:48,12:48] = True 

test = gray_image_matrix.copy()

test[12:48,12:48] = None # mauvais type, à voir comment changer ça
test_image = Image.fromarray(test)
test_image.show()



p1 = test[8:17,8:17] # patch 9
p1_image = Image.fromarray(p1)
p1_image.show()
p2 = test[44:53,44:53] # patch 9
p2_image = Image.fromarray(p2)
p2_image.show()

new_matrix = gray_image_matrix.copy()

D = choose_q(target_region_mask, p1, gray_image_matrix)
#m = min(D, key=D.get) # renvoie la clé de la valeur minimale
#mat = gray_image_matrix-omega
#m_matrix = mat[m[0]:m[0]+9,m[1]:m[1]+9]
#print(m)

#print(m_matrix)
#print(m_matrix.shape)
print(p1)
new_matrix[8:17,8:17] = D

D2 = choose_q(target_region_mask, p2, gray_image_matrix)
m2 = min(D2, key=D2.get) # renvoie la clé de la valeur minimale
#mat = gray_image_matrix-omega
#m2_matrix = mat[m2[0]:m2[0]+9,m2[1]:m2[1]+9]
#print(m2)
#print(m2_matrix)
#print(m2_matrix.shape)
print(p2)
new_matrix[44:53,44:53] = D2
new_matrix_image = Image.fromarray(new_matrix)
new_matrix_image.show()