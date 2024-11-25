from PIL import Image
import numpy as np
from skimage.util import view_as_windows
import luciano_final_functions as lf
import time
from scipy import signal


def calcul_dist_color(p,q, p_mask):
   
    if (p.shape[0],p.shape[1]) != (q.shape[0],q.shape[1]):
        print("p.shape:",p.shape)
        print("q.shape:",q.shape)
        raise ValueError("Les deux patchs n'ont pas la même taille")
    
    p_masked = p*p_mask
    q_masked = q*p_mask
    # Cast to float64 to prevent overflow
    p_masked = p_masked.astype(np.float64)
    q_masked = q_masked.astype(np.float64)
    diff = p_masked - q_masked
    sum_squared_diff = np.sum(diff ** 2, axis=(0,1))
    return np.sum(sum_squared_diff)


def choose_q(target_region_mask, front, p, p_mask, im, patch_size,margin = 60):
    D = {}
    
    source_region_mask = np.logical_not(target_region_mask)
    source_region_mask_size = source_region_mask.shape
    

    
    #Define the limits of the target region
    target_indices = np.argwhere(target_region_mask)
    min_x, min_y,min_z = target_indices.min(axis=0)
    max_x, max_y,min_z = target_indices.max(axis=0)
    

    #Define the limits of the source region with a margin
    min_x = max(0,min_x - margin)
    max_x = min(source_region_mask_size[0]- patch_size,max_x + margin + patch_size)
    min_y = max(0,min_y - margin)
    max_y = min(source_region_mask_size[1]-patch_size,max_y + margin + patch_size)

    for i in range(min_x,max_x-patch_size):
        for j in range(min_y,max_y-patch_size):
            q_mask = source_region_mask[i:i+patch_size,j:j+patch_size]
            valid_patch = np.all(q_mask)
            if valid_patch:
                q = im[i:i+patch_size,j:j+patch_size]
                d = calcul_dist_color(p, q, p_mask)
                D[(i,j)] = d
    
    # Find the patch with the minimum distance
    minimum_D = min(D, key=D.get)  # Returns the key of the minimum value
    q_opt = im[minimum_D[0]:minimum_D[0] + patch_size, minimum_D[1]:minimum_D[1] + patch_size]

    return q_opt



def update_target_region_mask(target_region_mask, selected_pixel, patch_size,im,im_size):
    
    updated_matrix = target_region_mask.copy()
    half_patch_size = patch_size // 2
    
    updated_matrix [max(selected_pixel[0] - half_patch_size, 0): min(selected_pixel[0] + half_patch_size + 1, im_size[0] - 1),max(selected_pixel[1] - half_patch_size, 0): min(selected_pixel[1] + half_patch_size + 1 , im_size[1] - 1)]= np.array([[[False] for i in range (patch_size)] for j in range(patch_size)])
    return updated_matrix


def neighbour_to_source_region(x, y, target_region_mask):
    source_region_mask = ~target_region_mask
    number_of_source_region_neighours = 0
    target_region_mask_size = target_region_mask.shape
    if x > 0 and x < target_region_mask_size[0]:
        if y > 0 and y < target_region_mask_size[1]:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0] + source_region_mask[x, y + 1,0]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x + 1, y,0] + source_region_mask[x, y + 1,0]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0]
    elif x == 0:
        if y > 0 and y < target_region_mask_size[1]:
            number_of_source_region_neighours += source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0] + source_region_mask[x, y + 1,0]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x + 1, y,0] + source_region_mask[x, y + 1,0]
        else:
            number_of_source_region_neighours += source_region_mask[x + 1, y,0] + source_region_mask[x, y - 1,0]
    else:
        if y > 0 and y < target_region_mask_size[1]:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x, y - 1,0] + source_region_mask[x, y + 1,0]
        elif y == 0:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x, y + 1,0]
        else:
            number_of_source_region_neighours += source_region_mask[x - 1, y,0] + source_region_mask[x, y - 1,0]
    return number_of_source_region_neighours > 0



def front_detection(im, target_region_mask,half_patch_size,im_size):
    
    #if target_region_mask.shape != im.shape:
    #    raise ValueError('target_region_mask and im must have the same shape')
    if np.all(target_region_mask == np.array([[[False] for i in range(im_size[1])] for j in range(im_size[0])])):
        return ("No target region")
    else : 
        front = np.array([[False for i in range(im_size[1])] for j in range(im_size[0])])
        #new_im = np.copy(im)
        for x in range(half_patch_size,im_size[0]-half_patch_size):
            for y in range(half_patch_size,im_size[1]-half_patch_size):
                if target_region_mask[x, y,0]:
                    front[x, y] = neighbour_to_source_region(x, y, target_region_mask)
        return front
  

def patch_search_compatible(target_region_mask, im, patch_size,margin = 70):

    im_matrix = np.array(im)
    new_matrix = im_matrix.copy()
    im_size = im.shape
   
    half_patch_size = patch_size // 2
    confidence_matrix = 1. - np.copy(target_region_mask)
    while target_region_mask.any():
        
        front = front_detection(new_matrix, target_region_mask,half_patch_size,im_size)
        
        pixel, confidence, data_term, priority = lf.pixel_with_max_priority(front, new_matrix, im,target_region_mask, confidence_matrix, im_size, patch_size)
        print(f"-- pixel : {pixel} | confidence : {confidence} | data term : {data_term} | priority : {priority}")
        if target_region_mask[pixel[0],pixel[1]] == np.array([True]):
            patch = new_matrix[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im_size[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im_size[1] - 1)]
            
            patch_mask = target_region_mask[max(pixel[0] - half_patch_size, 0) : min(pixel[0]+ half_patch_size + 1, im_size[0] - 1), max(pixel[1] - half_patch_size, 0) : min(pixel[1] + half_patch_size + 1, im_size[1] - 1)]
            # on met à False les valeurs de patch_mask qui correspondent à des valeurs de target_region_mask à True
            patch_mask = np.logical_not(patch_mask)
           
            q_patch = choose_q(target_region_mask, front, patch,  patch_mask, new_matrix, patch_size,margin)
            lf.show_patchs_chosen(pixel, patch, q_patch)

        
            new_matrix [max(pixel[0] - half_patch_size, 0):min(pixel[0]+ half_patch_size + 1, im_size[0] - 1),max(pixel[1] - half_patch_size, 0):min(pixel[1] + half_patch_size + 1, im_size[1] - 1)] = q_patch*(1-patch_mask) + patch*patch_mask
            #new_image = Image.fromarray(new_matrix)
            #new_image.show()
            
            confidence_matrix = lf.update_confidence(confidence_matrix, target_region_mask, pixel, confidence, patch_size, im_size)
            
            target_region_mask = update_target_region_mask(target_region_mask, pixel, patch_size,new_matrix,im_size)
             
    new_matrix = new_matrix.astype(np.uint8)        
    return new_matrix