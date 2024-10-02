import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

def front_detection(im, target_region_mask):
    if target_region_mask.shape != im.shape:
        raise ValueError('target_region_mask and im must have the same shape')
    if target_region_mask == np.array([[False for i in range(im.shape[0])] for j in range(im.shape[1])]):
        return ("No target region")
    else : 
        front = np.array([[False for i in range(im.shape[0])] for j in range(im.shape[1])])
        new_im = np.copy(im)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if target_region_mask[x, y]:
                    if not target_region_mask[x - 1, y] or not target_region_mask[x + 1, y] or not target_region_mask[x, y - 1] or not target_region_mask[x, y + 1]:
                        front[x, y] = True
        return front
               



