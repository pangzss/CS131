"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # flip the filter
    kernel_flipped = np.zeros(kernel.shape)
    for x in range(Hk):
        for y in range(Wk):
            kernel_flipped[x,y] = kernel[Hk-x-1,Wk-y-1]
            
    for i in range(Hi):
        for j in range(Wi):
            for m in range(Hk):
                for n in range(Wk):
                    x0,y0 = [int(i-(Hk-1)/2),int(j-(Wk-1)/2)]
                    x, y  = [x0+m,y0+n]
                    if x<0 or y<0 or x > Hi-1 or y > Wi-1:
                        out[i,j] = out[i,j]
                    else:
                        out[i,j] = out[i,j] + kernel_flipped[m,n]*image[x0+m,y0+n]
    
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    
    

    ### YOUR CODE HERE
    out[pad_height:pad_height+H,pad_width:pad_width+W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    kernel_flipped = np.zeros(kernel.shape)
    for x in range(Hk):
        for y in range(Wk):
            kernel_flipped[x,y] = kernel[Hk-x-1,Wk-y-1]

    ### YOUR CODE HERE
    pad_height,pad_width = int((Hk-1)/2),int((Wk-1)/2)
    im_padded = zero_pad(image, pad_height, pad_width)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.sum(np.multiply(kernel_flipped,im_padded[i:i+Hk,j:j+Wk]))
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    out = np.zeros((Hi, Wi))

    kernel_flipped = np.zeros(kernel.shape)
    for x in range(Hk):
        for y in range(Wk):
            kernel_flipped[x,y] = kernel[Hk-x-1,Wk-y-1]
    
    kernel_flipped_vec = kernel_flipped.reshape([1,Hk*Wk])
    ### YOUR CODE HERE
    pad_height,pad_width = int((Hk-1)/2),int((Wk-1)/2)
    im_padded = zero_pad(image, pad_height, pad_width)
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.dot(kernel_flipped_vec,im_padded[i:i+Hk,j:j+Wk].reshape([Hk*Wk,1]))
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    if Hg%2 == 0:
        Hg = Hg-1
    
    if Wg%2 == 0:
        Wg = Wg-1
        
    g = g[:Hg,:Wg]   
    g_vec = g.reshape([1,Hg*Wg])
    out = np.zeros((Hf, Wf))

    ### YOUR CODE HERE
    pad_height,pad_width = int((Hg-1)/2),int((Wg-1)/2)
    im_padded = zero_pad(f, pad_height, pad_width)
    print(im_padded.shape)
    for i in range(Hf):
        for j in range(Wf):
            out[i,j] = np.dot(g_vec,im_padded[i:i+Hg,j:j+Wg].reshape([Hg*Wg,1]))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_mean = np.mean(g)
    g = g-g_mean
    out = cross_correlation(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    
    if Hg%2 == 0:
        Hg = Hg-1
    
    if Wg%2 == 0:
        Wg = Wg-1
        
    g = g[:Hg,:Wg]   
    g_mean = np.mean(g)
    g_std = np.std(g)
    
    filter_vector = g.reshape([1,Hg*Wg])
    normalized_filter_vec = (g.reshape([1,Hg*Wg]) - g_mean)/g_std
    
    out = np.zeros((Hf, Wf))

    ### YOUR CODE HERE
    pad_height,pad_width = int((Hg-1)/2),int((Wg-1)/2)
    im_padded = zero_pad(f, pad_height, pad_width)

    for i in range(Hf):
        for j in range(Wf):
            patch_vector = im_padded[i:i+Hg,j:j+Wg].reshape([Hg*Wg,1])
            patch_mean = np.mean(patch_vector)
            patch_std = np.std(patch_vector)
            normalized_patch_vec = (patch_vector - patch_mean)/patch_std
            
            out[i,j] = np.dot(normalized_filter_vec,normalized_patch_vec)
    ### END YOUR CODE

    return out
