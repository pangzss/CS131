import math
import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """


    ### YOUR CODE HERE
    out = 0.5*np.power(image, 2)
   
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### YOUR CODE HERE
    out = color.rgb2grey(image)
    ### END YOUR CODE

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    if channel == 'R':
        k = 0
    elif channel == 'G':
        k = 1
    else:
        k = 2
        
    cImage = np.copy(image)        
    ### YOUR CODE HERE
    cImage[:,:,k] = 0
 

    out = cImage
    ### END YOUR CODE

    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    if channel == 'L':
        k = 0
    elif channel == 'A':
        k = 1
    else:
        k = 2
        
    lab = color.rgb2lab(image)
    

    ### YOUR CODE HERE
    out = lab[:,:,k]
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    if channel == 'H':
        k = 0
    elif channel == 'S':
        k = 1
    else:
        k = 2
        
    hsv = color.rgb2hsv(image)
 

    ### YOUR CODE HERE
    out = hsv[:,:,k]
    ### END YOUR CODE

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### YOUR CODE HERE
    s1 = image1.shape[1]
    s2 = image2.shape[1]
    print(s1,s2)
    out = np.concatenate((image1[:,:int(s1/2)],image2[:,int(s2/2):]),axis = 1)
    out = rgb_exclusion(out,channel1)
    out = rgb_exclusion(out,channel2)
    ### END YOUR CODE


    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    
    s   = image.shape[0:2]
    
    qTopLeft     = np.copy(image[:int(s[0]/2),:int(s[1]/2)])
    qTopRight    = np.copy(image[:int(s[0]/2),int(s[1]/2):])
    qBottomLeft  = np.copy(image[int(s[0]/2):,:int(s[1]/2)])
    qBottomRight = np.copy(image[int(s[0]/2):,int(s[1]/2):])
    
    qTopLeft     = rgb_exclusion(qTopLeft,'R')
    qTopRight    = dim_image(qTopRight)
    qBottomLeft  = np.power(qBottomLeft,2)
    qBottomRight = rgb_exclusion(qBottomRight,'R')
    
    outTop       =  np.concatenate((qTopLeft,qTopRight),axis = 1)
    outBottom    =  np.concatenate((qBottomLeft,qBottomRight),axis = 1)
 
    out = np.concatenate((outTop,outBottom),axis = 0)
    ### END YOUR CODE

    return out
