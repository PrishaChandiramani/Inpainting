from PIL import Image
import numpy as np
import luciano_functions as lf
import time


def calcul_dist(p,q, p_mask):
   # dans p, il peut y avoir des valeurs à None
   #p_mask[i,j] = True si il faut prendre la valeur du pixel dans p qu'elle n'est pas vide
   # si p_mask a que des valeurs à False, on ne fait rien, donc la distance est nulle, c'est à dire que le pixel q est parfait, problème à gérer, est qu'on commence à sum = -1 ?  
    sum = 0.0
    if p.shape != q.shape:
        raise ValueError("Les deux patchs n'ont pas la même taille")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if p_mask[i,j]:
                 # Cast to float64 to prevent overflow
 
                diff = float(p[i, j]) - float(q[i, j])
                sum += diff ** 2
    return sum

#def calcul_dist(p,q, p_mask):
#    return np.sum((p - q) ** 2 * p_mask)


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
    #print(D)
    minimum_D = min(D, key=D.get) # renvoie la clé de la valeur minimale
    q_opt = im[minimum_D[0]:minimum_D[0]+patch_size,minimum_D[1]:minimum_D[1]+patch_size]

    return q_opt 


def update_target_region_mask(target_region_mask, selected_pixel, patch_size,im):
    #print("in update_target_region_mask")
    updated_matrix = target_region_mask.copy()
    half_patch_size = patch_size // 2
    updated_matrix[selected_pixel[0],selected_pixel[1]] = False
    for x in range(max(selected_pixel[0] - half_patch_size, 0), min(selected_pixel[0] + half_patch_size + 1, im.shape[0] - 1)):
        for y in range(max(selected_pixel[1] - half_patch_size, 0), min(selected_pixel[1] + half_patch_size + 1, im.shape[1] - 1)):
            if target_region_mask[x, y]:
                updated_matrix[x, y] = False
    return updated_matrix



def neighbour_to_source_region(x, y, target_region_mask):
    number_of_source_region_neighours = 0
    if x > 0 and x < target_region_mask.shape[0]:
        if y > 0 and y < target_region_mask.shape[1]:
            number_of_source_region_neighours += target_region_mask[x - 1, y] + target_region_mask[x + 1, y] + target_region_mask[x, y - 1] + target_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += target_region_mask[x - 1, y] + target_region_mask[x + 1, y] + target_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += target_region_mask[x - 1, y] + target_region_mask[x + 1, y] + target_region_mask[x, y - 1]
    elif x == 0:
        if y > 0 and y < target_region_mask.shape[1]:
            number_of_source_region_neighours += target_region_mask[x + 1, y] + target_region_mask[x, y - 1] + target_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += target_region_mask[x + 1, y] + target_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += target_region_mask[x + 1, y] + target_region_mask[x, y - 1]
    else:
        if y > 0 and y < target_region_mask.shape[1]:
            number_of_source_region_neighours += target_region_mask[x - 1, y] + target_region_mask[x, y - 1] + target_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += target_region_mask[x - 1, y] + target_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += target_region_mask[x - 1, y] + target_region_mask[x, y - 1]
    return number_of_source_region_neighours > 0



def front_detection(im, target_region_mask):
    print("in front_detection")
    if target_region_mask.shape != im.shape:
        raise ValueError('target_region_mask and im must have the same shape')
    if np.all(target_region_mask == np.array([[False for i in range(im.shape[0])] for j in range(im.shape[1])])):
        return ("No target region")
    else : 
        front = np.array([[False for i in range(im.shape[0])] for j in range(im.shape[1])])
        new_im = np.copy(im)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if target_region_mask[x, y]:
                    front[x, y] = neighbour_to_source_region(x, y, target_region_mask)
        return front
    


def patch_search_compatible(target_region_mask, im, patch_size):
    # changer im en im_matrix pour être plus cohérent
    print("in patch_search_compatible")
    im_matrix = np.array(im)
    new_matrix = im_matrix.copy()
    new_matrix = new_matrix * (1- target_region_mask) + target_region_mask * 255
    half_patch_size = patch_size // 2
    confidence_matrix = 1. - np.copy(target_region_mask)
    while target_region_mask.any():
        print("in while loop")
        
        front = front_detection(new_matrix, target_region_mask)
        #lf.show_image(front, 'contour de la target region')
        pixel, confidence, data_term, priority = lf.pixel_with_max_priority(front, new_matrix, target_region_mask, confidence_matrix, im.shape, patch_size)
        print(f"pixel : {pixel} | confidence : {confidence} | data term : {data_term} | priority : {priority}")
        if target_region_mask[pixel[0],pixel[1]] == True:
            patch = new_matrix[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)]
            #print("patch_size:",patch_size)
            #print("patch:", patch)
            #print("patch.shape:",patch.shape)
            patch_mask = np.array([[True for i in range(patch_size)] for j in range(patch_size)])
            #print("patch_mask.shape:",patch_mask.shape)
            # on met à False les valeurs de patch_mask qui correspondent à des valeurs de target_region_mask à True
            #for k in range(max(pixel[0] - half_patch_size, 0),min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1)):
                #print("maxK: ",max(i - half_patch_size, 0))
                #print("minK: ",min(i+ half_patch_size + 1, im.shape[0] - 1))
                #print(k)
                #for l in range(max(pixel[1] - half_patch_size, 0),min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)):
                #    print("k,l",k,l)
                #    if target_region_mask[k,l]:
                #        patch_mask[k-max(pixel[0] - half_patch_size, 0),l-max(pixel[1] - half_patch_size, 0)] = False
                        #print("k',l':",k-max(i - half_patch_size, 0),l-max(j - half_patch_size, 0))

            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    global_i = max(pixel[0] - half_patch_size, 0) + i
                    global_j = max(pixel[1] - half_patch_size, 0) + j
                    if target_region_mask[global_i, global_j]:
                        patch_mask[i, j] = False

            #print("patch_mask:",patch_mask)
            q_patch = choose_q(target_region_mask, patch,  patch_mask, new_matrix, patch_size)
            #print("q_patch:",q_patch)
            #new_matrix [max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)] = q_patch
            new_matrix = update_matrix(q_patch, target_region_mask, pixel, half_patch_size, new_matrix)
            confidence_matrix = lf.update_confidence(confidence_matrix, target_region_mask, pixel, confidence, patch_size, im.shape)
            
            #lf.show_image(confidence_matrix, "matrice de confiance à jour")
            target_region_mask = update_target_region_mask(target_region_mask, pixel, patch_size,new_matrix)
            
            #print("target_region_mask:",target_region_mask)  
            #new_matrix_image = Image.fromarray(new_matrix)
            #new_matrix_image.show()  
    new_matrix = new_matrix.astype(np.uint8)        
    return new_matrix

def update_matrix(q_patch, target_region_mask, pixel, half_patch_size, new_matrix):
    updated_matrix = np.copy(new_matrix)
    for i in range(q_patch.shape[0]):
        global_i = max(pixel[0] - half_patch_size, 0) + i       
        for j in range(q_patch.shape[1]):
            global_j = max(pixel[1] - half_patch_size, 0) + j
            if target_region_mask[global_i,global_j]:
                updated_matrix[global_i,global_j] = q_patch[i,j]
    return updated_matrix



def patch_search_compatible_niterations(target_region_mask, im, patch_size,n):
    # changer im en im_matrix pour être plus cohérent
    pixel_list = []
    #print("in patch_search_compatible")
    im_matrix = np.array(im)
    new_matrix = im_matrix.copy()
    new_matrix = new_matrix * (1- target_region_mask) + target_region_mask * 255
    half_patch_size = patch_size // 2
    confidence_matrix = 1. - np.copy(target_region_mask)
    for i in range (n):
        #print("in for loop")
        front = front_detection(new_matrix, target_region_mask)
        pixel, confidence, data_term,priority = lf.pixel_with_max_priority(front, new_matrix, target_region_mask, confidence_matrix, im.shape, patch_size)
        print(f"pixel : {pixel} | confidence : {confidence} | data term : {data_term} | priority : {priority}")
        pixel_list.append(pixel)
        print("pixel :",pixel)
        if target_region_mask[pixel[0],pixel[1]] == True:
            patch = new_matrix[max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)]
            #printer ces indices
            print("encadrement : ",max(pixel[0] - half_patch_size, 0), min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1), max(pixel[1] - half_patch_size, 0), min(pixel[1] + half_patch_size + 1, im.shape[1] - 1))
            print("patch:", patch)
            patch_mask_target_region = target_region_mask[max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)]
            patch_mask = np.array([[False for i in range(patch_size)] for j in range(patch_size)])
            print("patch_mask.shape:",patch_mask.shape)

            for i in range(patch.shape[0]):
                for j in range(patch.shape[1]):
                    global_i = max(pixel[0] - half_patch_size, 0) + i
                    global_j = max(pixel[1] - half_patch_size, 0) + j
                    #print("global_i:",global_i)
                    #print("global_j:",global_j)
                    if not target_region_mask[global_i, global_j]:
                        patch_mask[i, j] = True

            print("patch_mask:",patch_mask)
            print("patch_mask_target_region: ", patch_mask_target_region)
            q_patch = choose_q(target_region_mask, patch,  patch_mask, new_matrix, patch_size)
            #print("q_patch:",q_patch)
            #new_matrix [max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)] = q_patch
            new_matrix = update_matrix(q_patch, target_region_mask, pixel, half_patch_size, new_matrix)
            confidence_matrix = lf.update_confidence(confidence_matrix, target_region_mask, pixel, confidence, patch_size, im.shape)
            target_region_mask = update_target_region_mask(target_region_mask, pixel, patch_size,new_matrix)
            #print("target_region_mask:",target_region_mask) 
            #print("new_matrix:",new_matrix) 
            new_matrix = new_matrix.astype(np.uint8)
            new_matrix_image = Image.fromarray(new_matrix)
            target_region_mask_matrix = target_region_mask * 255
            target_region_mask_matrix = target_region_mask_matrix.astype(np.uint8)
            #time.sleep(1)
            #new_matrix_image.show()
            #time.sleep(1)
            target_region_mask_image = Image.fromarray(target_region_mask_matrix)
            #target_region_mask_image.show()  
    #print(pixel_list)    
    return new_matrix