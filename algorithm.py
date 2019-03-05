# @author: Abhishek Jhoree
# @email: ajhoree@uoguelph.ca
# @ID: 0986820
# CIS *4720 Image Processing
# Assignment 2 Fire Detection algorithm

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

# Function to read in a gray scale image and return a 2d array
# @param fname[String] The name of the image
# @return [np Array] np array of the image
def imread_gray(fname):
    img = PIL.Image.open(fname)
    return np.asarray(img)

# Function to read in a color image and return as an 8-bit 2D array
# @param fname [String] the name of the file to open
# @return imCr
# @return imCg
# @return imCb
# @return imrgb
def imread_colour(fname):
    img = PIL.Image.open(fname)
    imgRGB = np.asarray(img)
    imCr = imgRGB[:,:,0]
    imCg = imgRGB[:,:,1]
    imCb = imgRGB[:,:,2]
    return imCr, imCg, imCb, imgRGB

# Saves an image locally
def imwrite_gray(fname,img):

    from scipy.misc import toimage
    toimage(img).save(fname + "RESTORED.jpg")
    toimage(img).show()

    img_uint8 = np.uint8(img)
    imgSv = PIL.Image.fromarray(img_uint8,'L')
    imgSv.save(fname +  "RESTORED.jpg")

# Function to convert an image to YcbCr spectrum using opencv
# @param img_name[String] Name of the image
# @return imgYCC [2D array] an array of the image in ycbcr spectrum
def rgcYcbcr(image):
    img = image.copy()
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
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

# finds the max rgc values
def max_rgb_filter(image):
	# split the image into its BGR components
	(B, G, R) = cv2.split(image)
 
	# find the maximum pixel intensity values for each
	# (x, y)-coordinate,, then set all pixel values less
	# than M to zero
	M = np.maximum(np.maximum(R, G), B)
	R[R < M] = 0
	G[G < M] = 0
	B[B < M] = 0
 
	# merge the channels back together and return the image
	return cv2.merge([B, G, R])

# makes a 3d plot of the image
def show_3d_plot(img, hsv_img):
    r, g, b = cv2.split(img, hsv_img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

    h, s, v = cv2.split(hsv_img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def sobel_function(image):
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    
    src = cv2.GaussianBlur(image, (3, 3), 0)
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv2.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    
    cv2.imshow(window_name, grad)
    cv2.waitKey(0)

def detect_fire(image, img_mode = 'hsv'):
    
    hsv_color1 = np.asarray([0, 0, 255])   # white!
    hsv_color2 = np.asarray([[30, 255, 255]]) # orange ish color
    blur = cv2.GaussianBlur(image, (21, 21), 0)
    # hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # blur=image

    # lower = [18, 50, 50]
    # upper = [35, 255, 255]
    if img_mode == 'hsv':
        # lower = [18, 50, 50]
        # upper = [35, 255, 255]
        lower = [20, 35, 220]
        upper = [35, 255, 255]
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    else:
        # lower = [116, 50, 50]
        # upper = [130, 130, 130]
        lower = [100, 50, 128]
        upper = [255, 128, 200]
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2YCR_CB)
    
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)
    
    plt.imshow(mask, cmap='gray')   # this colormap will display in black / white
    plt.show()
    result = cv2.bitwise_and(image, image, mask=mask)
    plt.imshow(result)
    plt.show()
 
    output = cv2.bitwise_and(image, hsv, mask=mask)
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
    print('Image size: ', image.size)
    im_size = float(image.size) * 0.25
    ratio = float(cv2.countNonZero(mask))/float(im_size)
    print('pixel percentage:', np.round(ratio*100, 2))

    

def main():
    if len(sys.argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    
    img_name = sys.argv[1]

    img2 = imread_colour(img_name)

    img = cv2.imread(img_name)
    hsv_img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_RGB2HSV)
 
    ycbcr_image = rgcYcbcr(img)

    

    detect_fire(hsv_img)
    detect_fire(ycbcr_image, 'ycc')
    # sobel_function(img)
    # spectrum = max_rgb_filter(img)


    # show_3d_plot(img, hsv_img)

if __name__ == '__main__':
    main()
