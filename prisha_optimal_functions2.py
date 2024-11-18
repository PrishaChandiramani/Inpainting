from PIL import Image
import numpy as np
from skimage.util import view_as_windows
import luciano_optimal_functions2 as lf
import time


def calcul_dist(p,q, p_mask):
   # dans p, il peut y avoir des valeurs à None
   #p_mask[i,j] = True si il faut prendre la valeur du pixel dans p qu'elle n'est pas vide
   # si p_mask a que des valeurs à False, on ne fait rien, donc la distance est nulle, c'est à dire que le pixel q est parfait, problème à gérer, est qu'on commence à sum = -1 ?  

    if p.shape != q.shape:
        raise ValueError("Les deux patchs n'ont pas la même taille")
    p_masked = p*p_mask
    q_masked = q*p_mask
    # Cast to float64 to prevent overflow
    diff = p_masked - q_masked
    sum = np.sum(diff ** 2)
    return sum

def calcul_dist_color(p,q, p_mask):
   # dans p, il peut y avoir des valeurs à None
   #p_mask[i,j] = True si il faut prendre la valeur du pixel dans p qu'elle n'est pas vide
   # si p_mask a que des valeurs à False, on ne fait rien, donc la distance est nulle, c'est à dire que le pixel q est parfait, problème à gérer, est qu'on commence à sum = -1 ?  

    if p.shape != q.shape:
        raise ValueError("Les deux patchs n'ont pas la même taille")
    p_masked = p*p_mask[...,np.newaxis]
    q_masked = q*p_mask[...,np.newaxis]
    # Cast to float64 to prevent overflow
    diff = p_masked - q_masked
    sum = np.sum(diff ** 2, axis=(0,1))
    return np.sum(sum)

#memory optimization
def choose_q_memory_optimization(target_region_mask, p,p_mask,im, patch_size):
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
            if np.any(1- q_mask):
                valid_patch = False #on peut mettre un break après pour minimiser le nombre d'itérations
                if valid_patch:
                    q = im[i:i+patch_size,j:j+patch_size]
                    d = calcul_dist(p,q,p_mask)
                    D[(i,j)]=d
    #print(D)
    minimum_D = min(D, key=D.get) # renvoie la clé de la valeur minimale
    q_opt = im[minimum_D[0]:minimum_D[0]+patch_size,minimum_D[1]:minimum_D[1]+patch_size]

    return q_opt 

def choose_q_original(target_region_mask, front, p, p_mask, im, patch_size):
    D = {}
    margin = 20
    source_region_mask = np.logical_not(target_region_mask)
    print("source_region_mask.shape:", source_region_mask.shape)

    # Extract patches from the image and the mask
    patches = view_as_windows(im, (patch_size, patch_size))
    mask_patches = view_as_windows(source_region_mask, (patch_size, patch_size))
    
    # Flatten the patches for easier processing
    patches = patches.reshape(-1, patch_size, patch_size)
    mask_patches = mask_patches.reshape(-1, patch_size, patch_size)
    
    # Filter valid patches
    for idx, mask in enumerate(mask_patches):
        if np.all(mask):
            patch = patches[idx]
            d = calcul_dist(p, patch, p_mask)
            i, j = np.unravel_index(idx, (source_region_mask.shape[0] - patch_size + 1, source_region_mask.shape[1] - patch_size + 1))
            D[(i , j )] = d
    #print(D)
    # Find the patch with the minimum distance
    minimum_D = min(D, key=D.get)  # Returns the key of the minimum value
    q_opt = im[minimum_D[0]:minimum_D[0] + patch_size, minimum_D[1]:minimum_D[1] + patch_size]

    return q_opt

# time optimization
def choose_q(target_region_mask, front, p, p_mask, im, patch_size):
    D = {}
    margin = 20
    source_region_mask = np.logical_not(target_region_mask)
    print("source_region_mask.shape:", source_region_mask.shape)

    #Define the limits of the target region
    target_indices = np.argwhere(target_region_mask)
    min_x, min_y = target_indices.min(axis=0)
    max_x, max_y = target_indices.max(axis=0)

    #Define the limits of the source region with a margin
    min_x = max(0,min_x - margin)
    max_x = min(source_region_mask.shape[0],max_x + margin + patch_size)
    min_y = max(0,min_y - margin)
    max_y = min(source_region_mask.shape[1],max_y + margin + patch_size)

    #Extract the source region from the image
    source_region = im[min_x:max_x, min_y:max_y]
    source_region_mask = source_region_mask[min_x:max_x, min_y:max_y]


    
    # Extract patches from the image and the mask
    patches = view_as_windows(source_region, (patch_size, patch_size))
    mask_patches = view_as_windows(source_region_mask, (patch_size, patch_size))
    
    # Flatten the patches for easier processing
    patches = patches.reshape(-1, patch_size, patch_size)
    mask_patches = mask_patches.reshape(-1, patch_size, patch_size)
    
    # Filter valid patches
    for idx, mask in enumerate(mask_patches):
        if np.all(mask):
            patch = patches[idx]
            d = calcul_dist(p, patch, p_mask)
            i, j = np.unravel_index(idx, (source_region_mask.shape[0] - patch_size + 1, source_region_mask.shape[1] - patch_size + 1))
            D[(i + min_x, j + min_y)] = d
    #print(D)
    # Find the patch with the minimum distance
    minimum_D = min(D, key=D.get)  # Returns the key of the minimum value
    q_opt = im[minimum_D[0]:minimum_D[0] + patch_size, minimum_D[1]:minimum_D[1] + patch_size]

    return q_opt



def update_target_region_mask(target_region_mask, selected_pixel, patch_size,im):
    #print("in update_target_region_mask")
    updated_matrix = target_region_mask.copy()
    half_patch_size = patch_size // 2
    #updated_matrix[selected_pixel[0],selected_pixel[1]] = False
    updated_matrix [max(selected_pixel[0] - half_patch_size, 0): min(selected_pixel[0] + half_patch_size + 1, im.shape[0] - 1),max(selected_pixel[1] - half_patch_size, 0): min(selected_pixel[1] + half_patch_size + 1 , im.shape[1] - 1)]= np.array([[False for i in range (patch_size)] for j in range(patch_size)])
    return updated_matrix


def neighbour_to_source_region(x, y, target_region_mask):
    source_region_mask = 1. - target_region_mask
    number_of_source_region_neighours = 0
    if x > 0 and x < target_region_mask.shape[0]:
        if y > 0 and y < target_region_mask.shape[1]:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x + 1, y] + source_region_mask[x, y - 1]
    elif x == 0:
        if y > 0 and y < target_region_mask.shape[1]:
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += source_region_mask[x + 1, y] + source_region_mask[x, y - 1]
    else:
        if y > 0 and y < target_region_mask.shape[1]:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y - 1] + source_region_mask[x, y + 1]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y + 1]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y] + source_region_mask[x, y - 1]
    return number_of_source_region_neighours > 0



def front_detection(im, target_region_mask):
    # je vois pas comment optimiser celle-ci
    print("in front_detection")
    if target_region_mask.shape != im.shape:
        raise ValueError('target_region_mask and im must have the same shape')
    if np.all(target_region_mask == np.array([[False for i in range(im.shape[1])] for j in range(im.shape[0])])):
        return ("No target region")
    else : 
        front = np.array([[False for i in range(im.shape[1])] for j in range(im.shape[0])])
        new_im = np.copy(im)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if target_region_mask[x, y]:
                    front[x, y] = neighbour_to_source_region(x, y, target_region_mask)
        return front
    
def update_matrix(q_patch, target_region_mask, pixel, half_patch_size, new_matrix):
    updated_matrix = np.copy(new_matrix)
    for i in range(q_patch.shape[0]):
        global_i = max(pixel[0] - half_patch_size, 0) + i       
        for j in range(q_patch.shape[1]):
            global_j = max(pixel[1] - half_patch_size, 0) + j
            if target_region_mask[global_i,global_j]:
                updated_matrix[global_i,global_j] = q_patch[i,j]
    return updated_matrix


def patch_search_compatible(target_region_mask, im, patch_size):

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
        pixel, confidence, data_term, priority = lf.pixel_with_max_priority(front, new_matrix, im,target_region_mask, confidence_matrix, im.shape, patch_size)
        print(f"pixel : {pixel} | confidence : {confidence} | data term : {data_term} | priority : {priority}")
        if target_region_mask[pixel[0],pixel[1]] == True:
            patch = new_matrix[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)]
            #print("patch_size:",patch_size)
            #print("patch:", patch)
            #print("patch.shape:",patch.shape)
            patch_mask = target_region_mask[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)]
            # on met à False les valeurs de patch_mask qui correspondent à des valeurs de target_region_mask à True
            patch_mask = np.logical_not(patch_mask)
            #print("patch_mask.shape:",patch_mask.shape)
            #print("patch_mask:",patch_mask)
            q_patch = choose_q(target_region_mask, front, patch,  patch_mask, new_matrix, patch_size)
            #print("q_patch:",q_patch)
            new_matrix [max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im.shape[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im.shape[1] - 1)] = q_patch
            #new_matrix = update_matrix(q_patch, target_region_mask, pixel, half_patch_size, new_matrix)
            confidence_matrix = lf.update_confidence(confidence_matrix, target_region_mask, pixel, confidence, patch_size, im.shape)
            
            #lf.show_image(confidence_matrix, "matrice de confiance à jour")
            target_region_mask = update_target_region_mask(target_region_mask, pixel, patch_size,new_matrix)
            
            #print("target_region_mask:",target_region_mask)  
            #new_matrix_image = Image.fromarray(new_matrix)
            #new_matrix_image.show()  
    new_matrix = new_matrix.astype(np.uint8)        
    return new_matrix