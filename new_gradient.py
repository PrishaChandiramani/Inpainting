import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal

def region3x3(x, y, img, target_region_mask, depth):
    size_x = img.shape[0]
    size_y = img.shape[1]

    xmin = int(x - 1)
    xmax = int(x + 1)
    ymin = int(y - 1)
    ymax = int(y + 1)
    print(f"xmin : {xmin}, xmax : {xmax}, ymin : {ymin}, ymax : {ymax}")
    local_target_region_mask = np.array([[False for i in range(3)] for j in range(3)])
    if xmin < 0:
        if ymin < 0:
            local_target_region_mask[1:, 1:] = target_region_mask[0:(xmax + 1), 0:(ymax + 1)]
        elif ymax >= size_y:
            local_target_region_mask[1:,:2] = target_region_mask[0:(xmax + 1), ymin:]
        else:
            local_target_region_mask[1:,:] = target_region_mask[0:(xmax + 1), ymin:(ymax + 1)]
    elif xmax >= size_x:
        if ymin < 0:
            local_target_region_mask[:2, 1:] = target_region_mask[xmin:, 0:(ymax + 1)]
        elif ymax >= size_y:
            local_target_region_mask[:2,:2] = target_region_mask[xmin:, ymin:]
        else:
            local_target_region_mask[:2,:] = target_region_mask[xmin:(xmax+1), ymin:(ymax + 1)]

    else:
        if ymin < 0:
            local_target_region_mask[:, 1:] = target_region_mask[xmin:(xmax + 1), 0:(ymax + 1)]
        elif ymax >= size_y:
            local_target_region_mask[:,:2] = target_region_mask[xmin:(xmax + 1), ymin:]
        else:
            local_target_region_mask[:,:] = target_region_mask[xmin:(xmax + 1), ymin:(ymax + 1)]
        
    print(f"local target region mask : {local_target_region_mask}")
    if np.any(local_target_region_mask) and depth > 0:
        
        x_matrix = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        y_matrix = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

        #print(f"x_matrix shape : {x_matrix.shape}, y_matrix shape : {y_matrix.shape}, local_target_region_mask shape : {local_target_region_mask.shape}")
        new_x = int(x + np.sign(np.sum(x_matrix * local_target_region_mask)))
        new_y = int(y + np.sign(np.sum(y_matrix * local_target_region_mask)))
        return region3x3(new_x, new_y, img, target_region_mask, depth-1)
    else:
        result = np.array([[img[xmin, ymin], img[xmin, y], img[xmin, ymax]], [img[x, ymin], img[x, y], img[x, ymax]], [img[xmax, ymin], img[xmax, y], img[xmax, ymax]]])
        return result
    
def new_gradient(pixel, image, target_region_mask):
    gradient = [0. , 0.]
    x, y = pixel[0], pixel[1]

    pixel_region = region3x3(x, y, image, target_region_mask, 5)
    
    gradient_core_x = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_core_y = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient[0] = np.sum(gradient_core_x * pixel_region)
    gradient[1] = np.sum(gradient_core_y * pixel_region)

    return gradient

def new_orthogonal_front_vector(pixel, target_region_mask):
    gradient = [0. , 0.]
    x, y = pixel[0], pixel[1]

    pixel_region = region3x3(x, y, target_region_mask, target_region_mask, 0)
    
    gradient_core_x = 1/4 * np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_core_y = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient[0] = np.sum(gradient_core_x * pixel_region)
    gradient[1] = np.sum(gradient_core_y * pixel_region)

    return gradient

if __name__ == "__main__":
    
    test_image = np.zeros((10, 10))
    test_target_region_mask = np.copy(test_image)
    for i in range(10):
        for j in range(10):
            if i >= j:
                test_target_region_mask[i, j] = 1.


    #print(new_gradient([0, 0], test_image, []))
    
   
    result_test = region3x3(4, 3, test_image, test_target_region_mask)
    plt.imshow(test_target_region_mask, cmap='grey')
    plt.show()

    