'''
Ethan Tompkins 
Assignment 2
CPSC 4820
'''

import cv2 as cv
from cv2 import scaleAdd
from cv2 import displayOverlay
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.util import random_noise
import copy

# set question to 1, 2, 3 (ints) or 'all' (string)
question = 'all'

def display_image(img, title, gray:bool = False, colorbar:bool = False):
    if gray:
        im_show = plt.imshow(img, cmap='gray') 
    else:
        im_show = plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    if colorbar:
        plt.colorbar(im_show, orientation="vertical")
    plt.show()

if question == 1 or question == 'all':
    # read first image
    zebra = cv.imread("zebra.jpg", cv.IMREAD_COLOR)
    zebra_height = zebra.shape[0]
    zebra_width = zebra.shape[1]
    
    # ============================================================================
    # START 1.1 ------------------------------------------------------------------
    # ============================================================================
    # convert to grayscale
    fg = cv.cvtColor(zebra, cv.COLOR_BGR2GRAY)

    # display greyscale
    display_image(img=fg, title="fg by Danny Saad", gray=True, colorbar=True)
    # ============================================================================
    # END 1.1 --------------------------------------------------------------------
    # ============================================================================



    # ============================================================================
    # START 1.2 ------------------------------------------------------------------
    # ============================================================================
    
    # get separate color layers and display 
    b = copy.deepcopy(zebra)
    b[:,:,1] = 0
    b[:,:,2] = 0
    # display_image(img=cv.cvtColor(b, cv.COLOR_BGR2RGB), title="blue by Danny Saad")

    g = copy.deepcopy(zebra)
    g[:,:,0] = 0
    g[:,:,2] = 0
    # display_image(img=cv.cvtColor(g, cv.COLOR_BGR2RGB), title="green by Danny Saad")

    r = copy.deepcopy(zebra)
    r[:,:,0] = 0
    r[:,:,1] = 0
    # display_image(img=cv.cvtColor(r, cv.COLOR_BGR2RGB), title="red by Danny Saad")
    
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(cv.cvtColor(r, cv.COLOR_BGR2RGB))
    axarr[0].set_title("red by dsaad")
    axarr[1].imshow(cv.cvtColor(g, cv.COLOR_BGR2RGB))
    axarr[1].set_title("green by dsaad")
    axarr[2].imshow(cv.cvtColor(b, cv.COLOR_BGR2RGB))
    axarr[2].set_title("blue by dsaad")
    plt.show()
    # ============================================================================
    # END 1.2 --------------------------------------------------------------------
    # ============================================================================



    # ============================================================================
    # START 1.3 ------------------------------------------------------------------
    # ============================================================================
    fht = copy.deepcopy(fg)
    for y in range(zebra_height):
        for x in range(zebra_width):
            pixel = fht[y,x]
            if pixel < 50:
                pixel = 0
            elif pixel > 200:
                pixel = 255
            fht[y,x] = pixel
    display_image(img=fht, title="Hard Thresholded by Danny Saad", gray=True, colorbar=True)

    # difference: makes whites whiter and blacks blacker, but leaves everything else the same - makes the colors pop more
    # ============================================================================
    # END 1.3 --------------------------------------------------------------------
    # ============================================================================



    # ============================================================================
    # START 1.4 ------------------------------------------------------------------
    # ============================================================================
    #-------- a --------
    g1 = np.ones((3, 3), np.float32)/9
    g1_image = cv.filter2D(src=fg, kernel=g1, ddepth=-1)
    display_image(img=g1_image, title="G1 by Danny Saad (3x3 average)", gray=True)

    #-------- b --------
    g2 = np.ones((10, 10), np.float32)/100
    g2_image = cv.filter2D(src=fg, kernel=g2, ddepth=-1)
    display_image(img=g2_image, title="G2 by Danny Saad (10x10 average)", gray=True)

    #-------- c --------
    g3 = np.array([[1,   1.5,  1],
                   [0,   0,    0],
                   [-1, -1.5, -1]])

    g3_image = cv.filter2D(src=fg, kernel=g3, ddepth=-1)
    display_image(img=g3_image, title="G3 by Danny Saad (edge x)", gray=True)

    #-------- d --------
    g4 = np.array([[1,   0,   -1],
                   [1.5, 0, -1.5],
                   [1,   0,   -1]])

    g4_image = cv.filter2D(src=fg, kernel=g4, ddepth=-1)
    display_image(img=g4_image, title="G4 by Danny Saad (edge y)", gray=True)

    #-------- e --------
    g5_image = cv.filter2D(src=fg, kernel=g3, ddepth=-1)
    g5_image = cv.filter2D(src=g5_image, kernel=g4, ddepth=-1)
    display_image(img=g5_image, title="G5 by Danny Saad (both x and y)", gray=True)

    #-------- f --------
    g6 = np.zeros((5,5), np.float32)
    for i in range(5):
        for j in range (5):
            g6[i,j] = np.exp(-((i-2)**2 + (j-2)**2) / (2 * 1))
    k_g6 = 1 / g6.sum()
    g6 = g6 * k_g6
    
    g7 = np.zeros((5,5), np.float64)
    for i in range(5):
        for j in range (5):
            g7[i,j] = np.exp(-((i-2)**2 + (j-2)**2) / (2 * 0.01))
    k_g7 = 1 / g7.sum()
    g7 = g7 * k_g7
    
    g6_image = cv.filter2D(src=fg, kernel=g6, ddepth=-1)
    g7_image = cv.filter2D(src=fg, kernel=g7, ddepth=-1)
    display_image(img=g6_image, title="G6 by Danny Saad", gray=True)
    display_image(img=g7_image, title="G7 by Danny Saad", gray=True)
    # ============================================================================
    # END 1.4 --------------------------------------------------------------------
    # ============================================================================
    
    
    
if question == 2 or question == 'all':
    # ============================================================================
    # START 2.1 ------------------------------------------------------------------
    # ============================================================================
    traffic = cv.imread("Traffic.jpg", cv.IMREAD_COLOR)
    traffic_gray = cv.cvtColor(traffic, cv.COLOR_BGR2GRAY)

    traffic_height = traffic.shape[0]
    traffic_width = traffic.shape[1]

    # apply gaussian noise
    gaus_img = random_noise(traffic_gray, mode='gaussian', mean=0, var=0.1*1)
    gaus_img = (255*gaus_img).astype(np.uint8)
    display_image(img=gaus_img, title="Gaussian Image - No Filtering", gray=True)
    
    # apply gaussian blur
    gaus_blur = cv.GaussianBlur(gaus_img, (5,5), math.sqrt(0.1*255))
    display_image(img=gaus_blur, title="Gaussian Image - Gaussian Blur", gray=True)
    
    # apply Sobel-X
    sobelx5 = cv.Sobel(gaus_img, cv.CV_64F, 0, 1, ksize=5)
    display_image(img=sobelx5, title="Gaussian Image - Sobel X", gray=True)
    
    # apply Sobel-Y
    sobely5 = cv.Sobel(gaus_img, cv.CV_64F, 1, 0, ksize=5)
    display_image(img=sobely5, title="Gaussian Image - Sobel Y", gray=True)

    # apply median filtering
    median_img = cv.medianBlur(gaus_img, 5)
    display_image(img=median_img, title="Gaussian Image - Median Blur", gray=True)
    
    # apply LoG filtering
    log_image = cv.Laplacian(gaus_blur, cv.CV_64F)
    display_image(img=log_image, title="Gaussian Image - LoG", gray=True)
    #plot log
    # ============================================================================
    # END 2.1 --------------------------------------------------------------------
    # ============================================================================
    
    
    
    # ============================================================================
    # START 2.2 ------------------------------------------------------------------
    # ============================================================================
    # apply s&p noise
    sp_img = random_noise(traffic_gray, mode='s&p', amount=0.08)
    sp_img = (255*sp_img).astype(np.uint8)
    display_image(img=sp_img, title="S & P Image - No Filtering", gray=True)
    
    # apply gaussian blur
    gaus_blur = cv.GaussianBlur(sp_img, (5,5), math.sqrt(0.1*255))
    display_image(img=gaus_blur, title="S & P Image - Gaussian Blur", gray=True)
    
    # apply Sobel-X
    sobelx5 = cv.Sobel(sp_img, cv.CV_64F, 0, 1, ksize=5)
    display_image(img=sobelx5, title="S & P Image - Sobel X", gray=True)
    
    # apply Sobel-Y
    sobely5 = cv.Sobel(sp_img, cv.CV_64F, 1, 0, ksize=5)
    display_image(img=sobely5, title="S & P Image - Sobel Y", gray=True)

    # apply median filtering
    median_img = cv.medianBlur(sp_img, 5)
    display_image(img=median_img, title="S & P Image - Median Blur", gray=True)
    
    # apply LoG filtering
    log_image = cv.Laplacian(gaus_blur, cv.CV_64F)
    display_image(img=log_image, title="S & P Image - LoG", gray=True)
    # ============================================================================
    # END 2.2 --------------------------------------------------------------------
    # ============================================================================


    
if question == 3 or question== 'all':
    # ============================================================================
    # START 3.1 ------------------------------------------------------------------
    # ============================================================================
    zc = cv.imread("zebra-cheetah.jpg", cv.IMREAD_COLOR)
    zc_grey = cv.cvtColor(zc, cv.COLOR_BGR2GRAY)
    zc_height = zc.shape[0]
    zc_width = zc.shape[1]
    
    f_fft = cv.dft(np.float32(zc_grey),flags = cv.DFT_COMPLEX_OUTPUT)
    f_fft = np.fft.fftshift(f_fft)
    f_fft_mag = cv.magnitude(f_fft[:,:,0],f_fft[:,:,1])
    f_fft_phase = cv.phase(f_fft[:,:,0],f_fft[:,:,1])
    f_fft_fixed_mag = 20*np.log(np.abs(f_fft_mag))
    display_image(f_fft_fixed_mag, title="F FFT Magnitude", gray=True)
    display_image(f_fft_phase, title="F FFT Phase", gray=True)
    # ============================================================================
    # END 3.1 --------------------------------------------------------------------
    # ============================================================================



    # ============================================================================    
    # START 3.2 ------------------------------------------------------------------
    # ============================================================================
    zc_scaled = zc_grey * 0.9

    def fs(a, T):
        max_pixel_value = np.amax(zc_scaled)
        # max_pixel_value = 255
        temp_img = copy.deepcopy(zc_scaled)
        for i in range(zc_height):
            for j in range(zc_width):
                f = temp_img[i,j]
                f = a * f + (1-a) * max_pixel_value * math.cos(2*math.pi*i / T) * math.cos(2*math.pi*j / T)
                temp_img[i,j] = f
        return temp_img

    for a in [0.9, 0.5, 0.1]:
        for T in [10, 20]:
            current_img = fs(a, T)
            current_img_fft = np.fft.fft2(current_img)
            current_img_fft = np.fft.fftshift(current_img_fft)
            current_img_fft = 20*np.log(np.abs(current_img_fft))
            display_image(current_img_fft, title="a = " + str(a) + " T = " + str(T), gray=True)
    # ============================================================================
    # END 3.2 --------------------------------------------------------------------
    # ============================================================================



    # ============================================================================
    # START 3.3 ------------------------------------------------------------------
    # ============================================================================
    fn = random_noise(zc_grey, mode='s&p',amount=0.05)
    fn = (255*fn).astype(np.uint8)
    fn_fft = cv.dft(np.float32(fn),flags = cv.DFT_COMPLEX_OUTPUT)
    fn_fft = np.fft.fftshift(fn_fft)
    fn_fft_mag = cv.magnitude(fn_fft[:,:,0],fn_fft[:,:,1])
    fn_fft_phase = cv.phase(fn_fft[:,:,0],fn_fft[:,:,1])
    fn_fft_fixed_mag = 20*np.log(np.abs(fn_fft_mag))
    
    display_image(fn_fft_fixed_mag, title="FN FFT Magnitude (Noise)", gray=True)
    display_image(fn_fft_phase, title="FN FFT Phase (Noise)", gray=True)
    # ============================================================================
    # END 3.3 --------------------------------------------------------------------
    # ============================================================================



    # ============================================================================    
    # START 3.4 ------------------------------------------------------------------
    # ============================================================================
    fn_ifft = np.fft.ifftshift(fn_fft)
    fn_ifft = cv.idft(fn_ifft)
    fn_img = cv.magnitude(fn_ifft[:,:,0],fn_ifft[:,:,1])
    display_image(fn_img, title="FN Reconstructed (Noise)", gray=True)
    # ============================================================================
    # END 3.4 --------------------------------------------------------------------
    # ============================================================================



    # ============================================================================
    # START 3.5 ------------------------------------------------------------------
    # ============================================================================    
    # create filter
    a = 0.17
    b = 0.1
    xmax = zc_width/2
    ymax = zc_height/2
    filter = np.zeros((zc_height, zc_width,2), np.float32)
    
    # verticle rectangle 
    for x in range(int(xmax*(1-a)), int(xmax*(1+a))):
        for y in range(int(ymax*(1-b)), int(ymax*(1+b))):
            filter[y,x] = 255
    
    # horizontal rectangle    
    for x in range(int(xmax*(1-b)), int(xmax*(1+b))):
        for y in range(int(ymax*(1-a)), int(ymax*(1+a))):
            filter[y,x] = 255
    
    filtered_fn = fn_fft * filter
    filtered_fn = np.fft.ifftshift(filtered_fn)
    filtered_img = cv.idft(filtered_fn)
    filtered_img = cv.magnitude(filtered_img[:,:,0],filtered_img[:,:,1])
    display_image(filtered_img, title="Low Pass Filtered Image (a = " + str(a) + ", b = " + str(b) + ")", gray=True)
    # ============================================================================
    # END 3.5 --------------------------------------------------------------------
    # ============================================================================

    


