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
def rgcYcbcr(img_name):
    img = cv2.imread(img_name)
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

