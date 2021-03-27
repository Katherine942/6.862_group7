
import cv2 
import numpy as np 
from skimage.util import * 

img = cv2.imread('pic_example.jpg')
cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()

noisy_img = random_noise(img , mode ='gaussian')
cv2.imshow('image',noisy_img)
cv2.waitKey()

cv2.imwrite('skimage_noisy.jpg',noisy_img)