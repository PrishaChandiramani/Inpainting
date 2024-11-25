import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal

def region3x3(x, y, img, target_region_mask, maxdepth):
    
    size_x = img.shape[0]
    size_y = img.shape[1]
    if x == 0 or y == 0 or x == size_x - 1 or y == size_y - 1:
        return np.zeros((3, 3))

    xmin, xmax = x - 1, x + 1
    ymin, ymax = y - 1, y + 1

    x_matrix = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    y_matrix = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

    local_target_region_mask = target_region_mask[xmin:xmax + 1, ymin:ymax + 1]

    depth = maxdepth
    while depth > 0 and np.any(local_target_region_mask):
        
        if x == 1 or x == size_x - 2 or y == 1 or y == size_y - 2:
            break

        x = int(x + np.sign(np.sum(x_matrix * local_target_region_mask)))
        y = int(y + np.sign(np.sum(y_matrix * local_target_region_mask)))
        
        xmin, xmax = x - 1, x + 1
        ymin, ymax = y - 1, y + 1

        local_target_region_mask = target_region_mask[xmin:xmax + 1, ymin:ymax + 1]
        depth -= 1

    result = img[xmin:xmax + 1, ymin:ymax + 1]
    return result
    
def new_gradient(pixel, image, target_region_mask):
    gradient = [0. , 0.]
    x, y = pixel[0], pixel[1]

    pixel_region = region3x3(x, y, image, target_region_mask, 3)
    
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
    if np.sqrt(gradient[0] ** 2 + gradient[1] ** 2) > 0:
        gradient = gradient / np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
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

    