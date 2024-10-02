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
    d = np.sum((p-q)**2)
    return d

def choose_q(omega, p):
    D={}
    psi = gray_image_matrix - omega #déterminer comment déterminer psi à partir de omega
    print(psi)
    psi_image = Image.fromarray(psi)
    psi_image.show()
    print("psi.shape:",psi.shape)
    for i in range(psi.shape[0]-9):
        for j in range(psi.shape[1]-9):
            q = psi[i:i+9,j:j+9]
            d = calcul_dist(p,q)
            D[(i,j)]=d
            #print("q.shape:",q.shape)
            #print("q:",q)
            #print("d:",d)
    return D  


omega = np.zeros((61,61))
omega[12:48,12:48] = gray_image_matrix[12:48,12:48] # patch 10
p1 = gray_image_matrix[12:21,12:21] # patch 9
p1_image = Image.fromarray(p1)
p1_image.show()
p2 = gray_image_matrix[30:39,30:39] # patch 9
p2_image = Image.fromarray(p2)
p2_image.show()

new_matrix = gray_image_matrix - omega

D = choose_q(omega, p1)
m = min(D, key=D.get) # renvoie la clé de la valeur minimale
mat = gray_image_matrix-omega
m_matrix = mat[m[0]:m[0]+9,m[1]:m[1]+9]
print(m)
print(m_matrix)
print(m_matrix.shape)
print(p1)
new_matrix[12:21,12:21] = m_matrix

D2 = choose_q(omega, p2)
m2 = min(D2, key=D2.get) # renvoie la clé de la valeur minimale
mat = gray_image_matrix-omega
m2_matrix = mat[m2[0]:m2[0]+9,m2[1]:m2[1]+9]
print(m2)
print(m2_matrix)
print(m2_matrix.shape)
print(p2)
new_matrix[30:39,30:39] = m2_matrix
new_matrix_image = Image.fromarray(new_matrix)
new_matrix_image.show()