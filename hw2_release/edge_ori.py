"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    kernel_flipped = np.zeros(kernel.shape)
    for x in range(Hk):
        for y in range(Wk):
            kernel_flipped[x,y] = kernel[Hk-x-1,Wk-y-1]
    
    kernel_flipped_vec = kernel_flipped.reshape([1,Hk*Wk])
    
    for i in range(Hi):
        for j in range(Wi):
            out[i,j] = np.dot(kernel_flipped_vec,padded[i:i+Hk,j:j+Wk].reshape([Hk*Wk,1]))
            
    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    k = (size-1)/2
    
    for i in range(size):
        for j in range(size):
            kernel[i,j] = np.exp(-( (i-k)**2 + (j-k)**2 )/(2*(sigma**2)) )
            
    kernel = 1/(2*np.pi*(sigma**2))*kernel

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([1.,0.,-1.],ndmin = 2)/2.
    out = conv(img, kernel)
 
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel = np.array([[1.],[0.] ,[-1.]], ndmin = 2)/2.
    out = conv(img, kernel)

    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = partial_x(img)
    Gy = partial_y(img)
    #print(Gx)
    #print(Gy)
    
    G = np.sqrt(Gx**2+Gy**2)
    theta = np.arctan2(Gy,Gx)*180/np.pi
    #print(theta)
    theta[(theta >= -180) & (theta < 0)] += 360
    #print(theta)
    
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    
    # edge padding
    pad_width0 =  1
    pad_width1 =  1
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    
    G_padded = np.pad(G, pad_width, mode='constant',constant_values = (0,0))
    theta_padded = np.pad(theta, pad_width, mode='constant',constant_values = (0,0))
    
    ### BEGIN YOUR CODE
    for i in range(H):
        for j in range(W):
            G_patch = G_padded[i:i+3,j:j+3]
            theta_patch = theta_padded[i:i+3,j:j+3]
            
        
            theta_current = theta[i,j]
            
            #indice = ((theta_patch == theta_current) | (theta_patch == theta_current + 180 - 2*(theta_current>=180)))
            
            #print("indice",indice)
            
            #pnDirection = G_patch[indice]
            
            if theta_current == 0 or theta_current == 180 or theta_current == 360:
                indice = ((1,2),(1,0))
            elif theta_current == 45 or theta_current == 225:
                indice = ((0,0),(2,2))
            elif theta_current == 90 or theta_current == 270:
                indice = ((0,1),(2,1))
            else:
                indice = ((0,2),(2,0))
           
            
            G_pnDirection = np.array([ 
                [  G_patch[indice[0]]  ],
                [  G_patch[indice[1]]  ] 
                                    ])
            '''
            if i == 244 and j == 411:
                print("G_patch",G_patch)
                print("theta_patch",theta_patch)
                print("G_pnDirection",G_pnDirection)
                print("G[i,j]",G[i,j]-G_pnDirection.max())
            '''
            '''
            if i == 1 and j == 2:
                print("G_patch",G_patch)
                print("theta_patch",theta_patch)
                print("G_pnDirection",G_pnDirection)
            '''
            if G[i,j] < G_pnDirection.max() or np.abs(G[i,j]- G_pnDirection.max()<1e-10):
                out[i,j] = 0
            else:
                out[i,j] = G[i,j]
                
           
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges = img>high
    weak_edges = (img<=high) & (img>low)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    new_edges = np.ones(edges.shape, dtype=np.bool)
    ### YOUR CODE HERE
   #while ~np.all(indices_pre == indices):
    while np.sum(new_edges) != 0:
        edges_pre = np.copy(edges)
        for index in indices:
            neighbors = get_neighbors(index[0], index[1], H, W)
            for n in neighbors:
                if weak_edges[n]:
                    edges[n] = True
        new_edges = edges - edges_pre
        indices = np.stack(np.nonzero(new_edges)).T
       
      
        
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G,theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges,weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    H,W = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for index in range(len(xs)):
        rho = xs[index] * cos_t + ys[index] * sin_t
        for x_index in range(num_thetas):
            y_index = np.rint(diag_len+rho[x_index])
            if y_index < 0:
                y_index = 0
            elif y_index > diag_len * 2.0 + 1:
                y_index = diag_len * 2.0 + 1
           
            accumulator[int(y_index),x_index] += 1
            
    ### END YOUR CODE

    return accumulator, rhos, thetas
