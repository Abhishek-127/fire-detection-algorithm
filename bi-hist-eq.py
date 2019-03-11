# @author: Abhishek Jhoree
# @email: ajhoree@uoguelph.ca

# first algorithm
# Implementation of the bi-histogram equalization

import numpy as np
from scipy.constants.constants import pi
from numpy.ma.core import exp
import math
import scipy.ndimage as nd
import pylab
import PIL
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib
import sys
from PIL import  Image
from scipy.misc import toimage
import scipy.misc
import time
import cv2

# Function to perform contrast enhancement using bi-histogram equalization
#   Ref(s):
#   Kim, Y.-T., "Contrast enhancement using brightness preserving bi-histogram
#   equalization", IEEE Trans. on Consumer Electronics, Vol.43, pp.1-8 (1997)
#
def histeqBI(im,nbr_bins=256):
    img = np.zeros(im.shape)

    gmin = 0 #im.min()
    gmax = 255 #im.max()
    
    # Calculate the image mean
    Xm = math.ceil(im.mean())
    imFLAT = im.flatten()
    
    # Find values <= Xm, and calculate the histogram (Eq.7)
#    Xlow = imFLAT.compress((imFLAT<=imFLAT.mean()).flat)
    Xlow = imFLAT.compress((imFLAT<=Xm).flat)
    piv1 = Xm
    HSTlow,bins = np.histogram(Xlow,piv1+1,(0,piv1),density=False)

    # Calculate the probability density function of the sub-histogram (Eq.9)
    HSTlowPDF = HSTlow / np.float32(Xlow.size)
    # Derive the cumulative histogram (Eq.11)
    cL = HSTlowPDF.cumsum() # cumulative distribution function
    
    #xr = np.arange(0,Xm+1,1)
    #pylab.plot(xr,cL)
    #pylab.show()
    
    # Find values > Xm, and calculate the histogram (Eq.8)
    Xupp = imFLAT.compress((imFLAT>Xm).flat)
    piv2 = 255-Xm
    HSTupp,bins = np.histogram(Xupp,piv2,(piv1+1,255),density=False)

    # Calculate the probability density function of the sub-histogram (Eq.10)
    HSTuppPDF = HSTupp / np.float32(Xupp.size)
    # Derive the cumulative histogram (Eq.12)
    cU = HSTuppPDF.cumsum() # cumulative distribution function
    
    #xr = np.arange(Xm+1,256,1)
    #pylab.plot(xr,cU)
    #pylab.show()

	# Histogram equalization for each of the sub-histograms
    fL = gmin + (Xm - gmin) * cL      #(Eq.13)
    fU = Xm+1 + (gmax - (Xm+1)) * cU  #(Eq.14)

    # Convert to 0-255
    fL = fL.astype('uint8')
    fU = fU.astype('uint8')
        
    # Merge the two histograms to make transformation easier
    fALL = np.concatenate((fL,fU))

    #xr = np.arange(0,256,1)
    #pylab.plot(xr,fALL)
    #pylab.show()
    
	# Transform the original image using the new cumulative histogram
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            img[i][j] = fALL[im[i][j]]
    
    # Return the modified image, and the cumulative histogram used to modify it
    return img, fALL

def SSIM(img_mat_1, img_mat_2):
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=np.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(np.float)
    img_mat_2=img_mat_2.astype(np.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=np.average(ssim_map)

    return index

def imwrite_gray(fname,img):

    from scipy.misc import toimage
    toimage(img).save(fname + "RESTORED.jpg")
    toimage(img).show()

    img_uint8 = np.uint8(img)
    imgSv = PIL.Image.fromarray(img_uint8,'L')
    imgSv.save(fname +  "RESTORED.jpg")

def imread_gray(fname):
    img = PIL.Image.open(fname)
    return np.asarray(img)

def plot_IMGhist(img,nbr_bins=256):    
    # the histogram of the data
    plt.hist(img.flatten(),nbr_bins,(0,nbr_bins-1))
    print("Flatten")
    print(img.flatten())
    print(len(img.flatten()) )
    plt.xlabel('Graylevels')
    plt.ylabel('No. Pixels')
    plt.title('Intensity Histogram')
    plt.grid(True)

    plt.show()

# Function to read in a grayscale image and return as an 8-bit 2D array
def imread_colour(fname):
    img = PIL.Image.open(fname)
    imgRGB = np.asarray(img)
    imCr = imgRGB[:,:,0]
    imCg = imgRGB[:,:,1]
    imCb = imgRGB[:,:,2]
    return imCr, imCg, imCb, imgRGB

def rgcYcbcr(img_name):
    img = cv2.imread(img_name)
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    print(imgYCC)
    cv2.imshow('my_image', imgYCC)
    cv2.waitKey(0)
    return imgYCC

def rgb_to_ycbcr(img):
    ycbcr_matrix = np.array(
    [[0.299, 0.587, 0.114],
    [-0.1687, -0.3313, 0.5],
    [0.5, -0.4187, -0.0813]]
    )
    
    ycbcr = img[3].dot(ycbcr_matrix.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def main():
    img_name = sys.argv[1]

    img = imread_colour(img_name)
    # img = cv2.imread(img_name)
    hsv_img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_RGB2HSV)
    # imwrite_gray(img_name, img[0])
    # new_image = rgb_to_ycbcr(img)
    
    # # plot_IMGhist(img[0])
    # # plot_IMGhist(new_image)
    # cv2.imshow('image',new_image)
    # cv2.waitKey(0)
    # cv2.imshow('hsv image', hsv_img)
    # cv2.waitKey(0)

    light_orange = (0, 0, 225)
    dark_orange = (255, 88, 35)

    lo_square = np.full((10, 10, 3), light_orange, dtype=np.uint8) / 255.0
    do_square = np.full((10, 10, 3), dark_orange, dtype=np.uint8) / 255.0
    plt.subplot(1, 2, 1)
    plt.imshow(hsv_to_rgb(do_square))
    plt.subplot(1, 2, 2)
    plt.imshow(hsv_to_rgb(lo_square))
    plt.show()

    mask = cv2.inRange(hsv_img, light_orange, dark_orange)
    result = cv2.bitwise_and(img[3], img[3], mask=mask)

    # plt.subplot(1, 2, 1)
    # plt.imshow(mask, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.imshow(result)
    # plt.show()
    hsv_color1 = np.asarray([0, 0, 255])   # white!
    hsv_color2 = np.asarray([[30, 255, 255]])   # yellow! note the order

    blur = cv2.GaussianBlur(cv2.imread(img_name), (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [18, 50, 50]
    upper = [35, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv_img, lower, upper)
    mask = cv2.inRange(hsv, lower, upper)
 
 
    output = cv2.bitwise_and(cv2.imread(img_name), hsv, mask=mask)
    # cv2.imshow("output", output)
    # plt.show()
    # plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
    # plt.show()
    # result = cv2.bitwise_and(img[3], img[3], mask=mask)
    # plt.imshow(result)
    # plt.show()
    red = cv2.countNonZero(mask)
    if red > 11000:
        print("Fire detected")
    else:
        print("No fire")


    print(red)


    plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
    plt.show()
    result = cv2.bitwise_and(img[3], img[3], mask=mask)
    plt.imshow(result)
    plt.show()

    spectrum = max_rgb_filter(cv2.imread(img_name))


    # show_3d_plot(img[3], hsv_img)
    

    
    

if __name__ == '__main__':
    main()


