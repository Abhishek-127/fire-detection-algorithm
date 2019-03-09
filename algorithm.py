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
from skimage import io, color

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

def rgb_to_lab(image):
    lab_image = color.rgb2lab(image)
    # cv2.imshow('my_image', lab_image)
    # cv2.waitKey(0)
    
    return lab_image

"""
Helper function to display an image to the screen
Will also save the image if you desire
@param image, The image to be displayed
@param image_name, [optional] the name to be displayed on the window
@param save [Boolean] whether or not you wish to save the image
@param saved_name [String] the name under which to store the image
    @pre must contain a valid extension e.g. jpg
"""
def display_image(image, image_name = 'Output', save = False, saved_name = ''):
    image = cv2.cvtColor(image, cv2.COLOR_YCR_CB2BGR) 
    cv2.imshow(image_name, image)
    cv2.waitKey(0)

    if save == True:
        cv2.imwrite(saved_name, image)



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


"""
Makes a 3d plot of the image
@param img rgb image
@param hsv_img image in hsv spectrum
"""
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

"""
Implementation of the algorithm for fire detection in the  hsv spectrum
This algorithm is based on the algorithm proposed by Wirth and Zaremba which
is based isolating the particular range correcponding to the fire in the image
A large set of images was examined to come up with the range of values used in this function
The algorithm has beed modified to also calculate the number of pixels caught by the masking process
This will later be used to  determine whether or not an image has a fire
@param image The image to be examined
@return red The number of red or 'fire' pixels detected in the image
"""
def detect_hsv_spectrum_fire(image):
    blur = cv2.GaussianBlur(image, (21, 21), 0)
    # lower = [20, 35, 100]
    # upper = [35, 255, 255]
    lower = [20, 150, 200]
    upper = [35, 255, 255]

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('maaaaaaa', mask)
    cv2.waitKey(0)
    red = cv2.countNonZero(mask)
    s = mask.shape[0] * mask.shape[1]
    per = float(red)/float(s)
    per = per * 100
    print('aaaaaaaaa', per)
    cv2.imwrite('mask.jpg', mask)

    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite('result.jpg', result)
    output = cv2.bitwise_and(image, hsv, mask=mask)
    out = cv2.countNonZero(output.flatten())
    size = image.shape[0] * image.shape[1]
    percentage = (float(out)/float(size)) * 100 
    print('perrrrrr = ', percentage)
    cv2.imwrite('output.jpg', output)

    display_image(output, 'hsv', True, 'ab.jpg')
    return percentage

def detect_fire(image, img_mode = 'hsv'):
    blur = cv2.GaussianBlur(image, (21, 21), 0)

    # lower = [18, 50, 50]
    # upper = [35, 255, 255]
    if img_mode == 'hsv':
        # lower = [18, 50, 50]
        # upper = [35, 255, 255]
        lower = [25, 35, 100]
        upper = [68, 255, 255]
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # hsv = image
    elif img_mode == 'lab':
        lower = [-40, -40, -40]
        upper = [80, 84, 90]
        # hsv = image
        #hsv = cv2.cvtColor(blur, cv2.)
    else:
        # lower = [68, 67, 66]
        # upper = [202, 202, 255]
        lower = [50, 50, 50]
        upper = [254, 253, 253]
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

    red = cv2.countNonZero(mask)
    if red > 11000:
        print("Fire detected")
    else:
        print("No fire")


    print(red)

    im_size = get_image_size(image) * 0.66
    ratio = float(red)/float(im_size)
    print('pixel percentage:', np.round(ratio*100, 2))
    return red

"""
Function to retrieve the size of an image
@param image The image whose size needs to be determined
@return size The size of the image in pixels
"""
def get_image_size(image):
    height = image.shape[0]
    width = image.shape[1]
    size = width * height
    print(size)
    return size

"""
Function to retrieve the mean of each color channel
Also retireves the standard deviation of each channel
This will help in the implementation of the algorithms
@param ycrcb_image
"""
def retrieve_mean_ycc(ycrcb_image):
    true_mean = cv2.mean(ycrcb_image)
    print(true_mean)
    Ymean = true_mean[0]
    CrMean = true_mean[1]
    CbMean = true_mean[2]

    # getting the standard dev
    means, stds = cv2.meanStdDev(ycrcb_image)
    cr_std = stds[1]
    
    return Ymean, CrMean, CbMean, cr_std

"""
Function that checks for the intensity of the y components as well as
the intensity of the cb components.
Only allow those regions with greater Y components than Cb
Blackens the rest of the pixels
@param image The image to be processed in the ycrcb spectrum
@return R1_image processed image with only pixels that satisfy that rule
"""
def rule1(image):
    R1_image = image.copy()
    height = image.shape[0]
    width = image.shape[1]

    pixel = 0

    for x in range(0, height):
        for y in range(0, width):
            if(image[x, y][0] > image[x, y][2]):
                R1_image[x, y] = image[x,y]
                pixel += 1
            else:
                R1_image[x,y] = [16, 128, 128]
    print('r1 fire pixels = ', float(pixel)/float(height*width)*100)
    display_image(R1_image, 'r1')
    return R1_image, pixel

"""
Flame region is the brightest region in the scene. We can assume that the
red chrominance (Cr) in th fire region will be more
This algorithm uses the values of the mean Y as well as mean Cr of the original image as
benchmark to filter out those pixels that do not satisfy the that rule
Rule 2 is applied on the pixels that have satisfied rule 1
@param image The image R1 from the previous step: This image has been filtered once and the 
    overall fire region has been indentified. 
    This will further remove the pixels in the non-fire region as they will not satisfy the condition
@overall_image The original image
@return R2_image Processed image after the filter has been applied
@return pixel The number of non-fire pixels in the image
"""
def rule2(image, overall_image):
    R2_image = image.copy()
    Ymean, crMean, c, s = retrieve_mean_ycc(overall_image.copy())

    height = image.shape[0]
    width = image.shape[1]
    pixel = 0
    for x in range(0, height):
        for y in range(0, width):
            if (image[x, y][0] > Ymean) and (image[x,y][1] > crMean):
                R2_image[x, y] = image[x, y]
                pixel = pixel + 1
            else:
                R2_image[x, y] = [16, 128, 128]
    display_image(R2_image, 'r2')
    print('r2 fire pixels = ', float(pixel)/float(height*width)*100)
    return R2_image, pixel

"""
Rule 3 is meant to identify the high temperature centre of the fire
Since the fire is the brightest at the centre, it will be almost white in color
This will help in identifying those regions
The idea is that Luminance Y will be greater that red chrominance and the blue Chrominance
Cb will be greater that Y. Based on these assumptions a filter will be applied on the image
to isolate the bright fire centre
@param image The image to be tested in YCrCb spectrum
@return R3_image The image will the filter applied to it
@return pixel the number of pixels that are super bright (satisfy this coondition)
"""
def rule3(image, rgb = ''):
    R3_image = image.copy()

    height = image.shape[0]
    width = image.shape[1]
    pixel = 0
    for x in range(0, height):
        for y in range(0, width):
            if(image[x, y][2] >= image[x, y][0] > image[x, y][1]):
                pixel = pixel + 1
                R3_image[x,y] = image[x, y]
            else:
                R3_image[x, y] = [16, 128, 128]

    display_image(R3_image, 'r3')
    print('r3 fire pixels = ', float(pixel)/float(height*width)*100)
    return R3_image,  pixel

"""
While the image is being segmented based on luminance, some of the white colored regions
like smoke and cloud will be segmented from the input image
Rule 4 is meant to overcome this. The texture of the fire  region will also be incorporated.
The fire and non fire like clouds have different textures.
The red chrominance will be less than the standard deviation of the red chrominance
@param image The resulting image after Rule 3
@param input_image The original image in YCrCb spectrum that will be used to get the statistical
    data required to apply the filter.
@return R4_image The image after the filter has been applied
@return pixel All the pixels that satisfy Rule 4
"""
def rule4(image, input_image):
    R4_image = image.copy()

    Ymean, CrMean, CbMean, cr_std = retrieve_mean_ycc(image)
    tau = 7.4

    height = image.shape[0]
    width = image.shape[1]
    pixel = 0

    for x in range(0, height):
        for y in range(0, width):
            if(image[x,y][1] < (tau * cr_std)):
                R4_image[x, y] = image[x, y]
                pixel += 1
            else:
                R4_image[x, y] = [16, 128, 128]

    display_image(R4_image, 'r4')
    print('r4 fire pixels = ', float(pixel)/float(height*width)*100)
    return R4_image, pixel

def lab_rule1(lab_image):
    R1 = lab_image.copy()
    R2 = lab_image.copy()
    R3 = lab_image.copy()
    R4 = lab_image.copy()
    Lmean, AMean, BMean, lab_std = retrieve_mean_ycc(lab_image)
    height = lab_image.shape[0]
    width = lab_image.shape[1]

    for x in range(0, height):
        for y in range(0, width):
            if(lab_image[x, y][0] >= Lmean):
                R1[x, y] = [1, 1, 1]
            else:
                R1[x, y] = [0, 0, 0]
            
            if(lab_image[x, y][1] >= AMean):
                R2[x, y] = [1, 1, 1]
            else:
                R2[x, y] = [0, 0, 0]

            if(lab_image[x, y][2] >= BMean):
                R3[x, y] = [1, 1, 1]
            else:
                R3[x, y] = [0, 0, 0]

            if(lab_image[x, y][2] >= lab_image[x, y][1]):
                R4[x, y] = [1, 1, 1]
            else:
                R4[x, y] = [0, 0, 0]
    display_image(R1)
    display_image(R2)
    display_image(R3)
    display_image(R4)
    return R1, R2, R3, R4


"""
Small helper function that helps retrieve the min and max values
from images. This will be used mainly to come up with benchmark values
using pictures with only fire in them
"""
def retrieve_min_max_values(image):

    YMax = np.max(image[0])
    YMin = np.min(image[0])
    cbMin = np.min(image[1])
    cbMax = np.max(image[1])
    crMin = np.min(image[2])
    crMax = np.max(image[2])
    print('Y Min = ', YMin)
    print('YMax = ', YMax)
    print('Cb Min = ', cbMin)
    print('cb Max = ', cbMax)
    print('cr Min = ', crMin)
    print('cr Max = ', crMax)

    return YMax, YMin, cbMin, cbMax, crMin, crMax

"""
Function to return the percentae of fire pixels based on the number of detected
    fire pixels and the original image
@param original The original image that is being processed
@param num_fire_pixels The number of fire pixels that have been detected based on the
    algorithm used
@return perentage The percentage of the image that is on Fire
"""
def return_percentage_fire(original_image, num_fire_pixels):
    if num_fire_pixels == 0:
        return 0

    num_pixels = get_image_size(original_image)
    percentage = (float(num_fire_pixels)/float(num_pixels)) * 100

    return percentage

"""
Function that analyzes the results after the set of rules to process YCrCb images
have been applied
"""
def analyze_ycc_results(p1, p2, p3, p4):
    is_fire = False
    if(p1 > 7 and p2 > 7):
        is_fire = True

    if(p3 > 2 and p4 > 2):
        is_fire = True

    return is_fire
"""
Analyzes the results after the flame detection algorithm usign the HSV spectrum have been
used. Based on the percentage of pixels that are fire pixels, it will determine whether or not
the image contains a fire
@param original_image The original image that is being processed
@param num_fire_pixels The number of pixels identified as fire in the image
@return is_fire Boolean indicating whether or not an image has fire
"""
def analyze_hsv_results(original_image, num_fire_pixels):
    is_fire = False
    percentage_fire = return_percentage_fire(original_image, num_fire_pixels)
    percentage_fire = percentage_fire / 0.667
    print('percentage fire hsv = ', percentage_fire)
    if percentage_fire >= 10:
        is_fire = True

    return is_fire

"""
Function that performs the final analysis based on the values returned by the various functions
It will apply some pre-defined set of benchmark scores that were obtained through experimentation
and those rules will determine whether or not a fire is present
@param hsv_percentage The percentage of fire pixels detected by the hsv algorithm
@param r1 Percentage of pixels satisfying rule 1 of YCbCr detection algorithm
@param r2 Percentage of pixels satisfying rule 2 of YCbCr detection algorithm
@param r3 Percentage of pixels satisfying rule 3 of YCbCr detection algorithm
@param r4 Percentage of pixels satisfying rule 4 of YCbCr detection algorithm
@param second_percentage Percentage of Fire pixels after YCbCr algoorithm
@param r1_i Image generated by rule 1
@param r2_i Image generated by rule 2
@param r3_i Image generated by rule 3
@param r4_i Image generated by rule 4
"""
def final_analysis(hsv_percentage, r1, r2, r3, r4, second_percentage, r1_i, r2_i, r3_i, r4_i):
    is_fire = False
    # first check is for fire with lots of smoke
    # The metric r3 as 3 was obtained through experimenting with
    #   forest fire images that typically have lots of smoke
    if(r3 > 4.0 and r1 > 0.0 and r2 > 0.0):
        is_fire = True
        
        # return is_fire
    
    # check for normal fires here
    if(hsv_percentage > 5 and second_percentage > 0 and r2 > 7 and r1 > 15):
        is_fire = True
    
    print('is there a fire = ', is_fire)


    return is_fire 

"""
Function that will handle the bulk of the task. This function handles all the function
calls to the different functions and will ultimately return the verdict on whether or not an
image has fire or not
@param image_name The name of the image that needs to be processed.

"""
def perform_fire_detection(image_name):
    is_fire = False
    hsv_fire = False
    # reads in the image
    image = cv2.imread(image_name)
    # converts the image in HSV for processing
    hsv_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    ycrcb_image = rgcYcbcr(image.copy())

    percentage_hsv = detect_hsv_spectrum_fire(hsv_image)
 
    print('percentage is ', percentage_hsv)

    ycrcb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2YCR_CB)

    R1, r1_pix = rule1(ycrcb_image)
    R2, r2_pix = rule2(R1, ycrcb_image)
    R3, r3_pix = rule3(ycrcb_image)
    R4, r4_pix = rule4(R3, ycrcb_image)

    r1_percentage = return_percentage_fire(R1, r1_pix)
    r2_percentage = return_percentage_fire(R2, r2_pix)
    r3_percentage = return_percentage_fire(R3, r3_pix)
    r4_percentage = return_percentage_fire(R4, r4_pix)

    print('wazzzaaa',r1_percentage, r2_percentage, r3_percentage, r4_percentage)
    i = cv2.cvtColor(R2.copy(), cv2.COLOR_YCR_CB2BGR)
    i = cv2.cvtColor(i.copy(), cv2.COLOR_BGR2HSV)
    second_percentage = detect_hsv_spectrum_fire(i)
    RFinal = cv2.add(R1, R3)
    is_fire = final_analysis(
        percentage_hsv,
        r1_percentage,
        r2_percentage,
        r3_percentage,
        r4_percentage,
        second_percentage,
        R1,
        R2,
        R3,
        R4
    )
    return is_fire

def main():
    if len(sys.argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    
    img_name = sys.argv[1]
    is_fire = perform_fire_detection(img_name)
    return 0

if __name__ == '__main__':
    main()
