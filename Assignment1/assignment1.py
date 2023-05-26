import math
import cv2 as cv
import numpy as np
import copy

# load image
img = cv.imread("bird.jpeg", cv.IMREAD_COLOR)


# store and show size of image
height = img.shape[0]
width = img.shape[1]
print(f"Image is {height} x {width}")


# display image
cv.imshow("original image", img)
cv.waitKey(0)
cv.destroyAllWindows()


# display blue, green, and red layers
b = copy.deepcopy(img)
b[:,:,1] = 0
b[:,:,2] = 0

g = copy.deepcopy(img)
g[:,:,0] = 0
g[:,:,2] = 0

r = copy.deepcopy(img)
r[:,:,0] = 0
r[:,:,1] = 0

cv.imshow("blue layer", b)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("green layer", g)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("red layer", r)
cv.waitKey(0)
cv.destroyAllWindows()


# resize image
resized_height = int(height * 0.5)
resized_width = int(width * 0.5)
resized_shape = (resized_width, resized_height)

resized_img = cv.resize(img, resized_shape, cv.INTER_AREA)

cv.imshow("resized image", resized_img)
cv.waitKey(0)
cv.destroyAllWindows()


# flip horizontally
img_flipped = img[:,::-1, :]

cv.imshow("flipped image", img_flipped)
cv.waitKey(0)
cv.destroyAllWindows()


# convert image to HSV
converted_colorspace = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cv.imshow("converted colorspace", converted_colorspace)
cv.waitKey(0)
cv.destroyAllWindows()


# remove green channel
no_green = copy.deepcopy(img) # create deepcopy of image, to avoid changing the original image
no_green[:,:,1] = np.zeros([height, width])

cv.imshow("no green", no_green)
cv.waitKey(0)
cv.destroyAllWindows()


# convert to greyscale, apply changes, scale
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

for y in range(height):
    for x in range(width):
        pixel = gray_img[y,x]
        pixel = math.log(1+pixel,10)
        pixel = 1+math.cos(pixel)
        gray_img[y,x] = pixel

gray_img = cv.normalize(gray_img, np.zeros((height,width)), 0, 255, cv.NORM_MINMAX)

cv.imshow("gray image", gray_img)
cv.waitKey(0)
cv.destroyAllWindows()


# perspecive transformations
t1 = np.array([[2,0,0],
               [0,1,0],
               [0,0,1]], 
              dtype='float32')
t2 = np.array([[math.sqrt(2), (-1)*math.sqrt(2), 0],
               [math.sqrt(2), math.sqrt(2), 0],
               [0,0,1]],
              dtype='float32')
t3 = np.array([[1,0.2,0],
               [0.2,1,0],
               [0,0,1]], 
              dtype='float32')
t4 = np.array([[1.1,0.1,0],
               [0.2,0.9,0],
               [0.1,0.2,1]],
              dtype='float32')


t1_img = cv.warpPerspective(img, t1, [width,height])
t2_img = cv.warpPerspective(img, t2, [width,height])
t3_img = cv.warpPerspective(img, t3, [width,height])
t4_img = cv.warpPerspective(img, t4, [width,height])

cv.imshow("t1 image", t1_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("t2 image", t2_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("t3 image", t3_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("t4 image", t4_img)
cv.waitKey(0)
cv.destroyAllWindows()
