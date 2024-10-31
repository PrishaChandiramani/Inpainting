import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy import signal

def region3x3(x, y, img, target_region_mask):
    size_x = img.shape[0]
    size_y = img.shape[1]

    xmin = int(x - 1 % size_x)
    xmax = int(x + 1 % size_x)
    ymin = int(y - 1 % size_y)
    ymax = int(y + 1 % size_y)
    local_target_region_mask = target_region_mask[xmin:xmax + 1, ymin:ymax + 1]

    if np.any(local_target_region_mask):
        
        x_matrix = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        y_matrix = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

        new_x = int(x + np.sign(np.sum(x_matrix * local_target_region_mask)))
        new_y = int(y + np.sign(np.sum(y_matrix * local_target_region_mask)))
        return region3x3(new_x, new_y, img, target_region_mask)

    else:
        result = np.array([[img[xmin, ymin], img[xmin, y], img[xmin, ymax]], [img[x, ymin], img[x, y], img[x, ymax]], [img[xmax, ymin], img[xmax, y], img[xmax, ymax]]])
        return result
    
def new_gradient(pixel, image, target_region_mask):
    gradient = [0. , 0.]
    x, y = pixel[0], pixel[1]

    pixel_region = region3x3(x, y, image, target_region_mask)
    
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

    