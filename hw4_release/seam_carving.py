"""
CS131 - Computer Vision: Foundations and Applications
Assignment 4
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 10/19/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import color


def energy_function(image):
    """Computes energy of the input image.

    For each pixel, we will sum the absolute value of the gradient in each direction.
    Don't forget to convert to grayscale first.

    Hint: Use np.gradient here

    Args:
        image: numpy array of shape (H, W, 3)

    Returns:
        out: numpy array of shape (H, W)
    """
    H, W, _ = image.shape
    out = np.zeros((H, W))
    gray_image = color.rgb2gray(image)

    ### YOUR CODE HERE
    G = np.gradient(gray_image)
    G = np.absolute(G)
    out = G[0]+G[1]
    ### END YOUR CODE

    return out


def compute_cost(image, energy, axis=1):
    """Computes optimal cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    In the case that energies are equal, choose the left-most path. Note that
    np.argmin returns the index of the first ocurring minimum of the specified
    axis.

    Make sure your code is vectorized because this function will be called a lot.
    You should only have one loop iterating through the rows.

    Args:
        image: not used for this function
               (this is to have a common interface with compute_forward_cost)
        energy: numpy array of shape (H, W)
        axis: compute cost in width (axis=1) or height (axis=0)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)
   
    # Initialization
    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    
    # iterate over rows
    for r in range(H-1):
        r += 1
        # leave out the first row
        # dynamic programming throuogh vectorized codes
        # shift the last row of cost matrix to the left
        # and the right respectively to find the minimums 
        # every three elements
        lastRow = np.copy(cost[r-1,:])
        lastRowShiftedR = np.insert(lastRow[:-1],0,lastRow[0])
        lastRowShiftedL = np.insert(lastRow[1:],-1,lastRow[-1])
        '''
        lastRowShiftedR = np.roll(lastRow, 1)
        lastRowShiftedL = np.roll(lastRow, -1)
        lastRowShiftedR[0] = lastRow[0]
        lastRowShiftedL[-1] = lastRow[-1]
        '''
        # combine the three rows obtained above into a 2D array
        # and then use np.amin to find minimums in each column
        # find the indices of the mins as well
        mat4FindMins = np.stack((lastRowShiftedR, lastRow,lastRowShiftedL))
        minVals = np.amin(mat4FindMins,axis=0)
        minIdces = np.argmin(mat4FindMins,axis=0)
        # add the current row with the newly found row of minimum costs,
        # and the result then replaces the current row
        
        cost[r,:] = energy[r,:] + minVals
        # use obtianed indices of mins to fill "paths"
        minIdces -= 1
        '''
        minIdces[minIdces == 0] = -1
        minIdces[minIdces == 1] = 0
        minIdces[minIdces == 2] = 1
        '''
        # shifting the row to the right and inserting a copy of the 
        # original row's first element into the array as the new head
        # may lead to a -1, which is incorrect and should be 0
        
        if minIdces[0] == -1:
            minIdces[0] = 0
       
        paths[r,:] = minIdces        
        
    ### END YOUR CODE

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    """Backtracks the paths map to find the seam ending at (H-1, end)

    To do that, we start at the bottom of the image on position (H-1, end), and we
    go up row by row by following the direction indicated by paths:
        - left (value -1)
        - middle (value 0)
        - right (value 1)

    Args:
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
        end: the seam ends at pixel (H, end)

    Returns:
        seam: np.array of indices of shape (H,). The path pixels are the (i, seam[i])
    """
    H, W = paths.shape
    # initialize with -1 to make sure that everything gets modified
    seam = - np.ones(H, dtype=np.int)

    # Initialization
    seam[H-1] = end

    ### YOUR CODE HERE
    for r in reversed(range(H)):
        if r != 0:
            seam[r-1] = seam[r] + paths[r,seam[r]]
   
    
    ### END YOUR CODE

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    """Remove a seam from the image.

    This function will be helpful for functions reduce and reduce_forward.

    Args:
        image: numpy array of shape (H, W, C) or shape (H, W)
        seam: numpy array of shape (H,) containing indices of the seam to remove

    Returns:
        out: numpy array of shape (H, W-1, C) or shape (H, W-1)
             make sure that `out` has same type as `image`
    """

    # Add extra dimension if 2D input
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    ### YOUR CODE HERE
    out = np.zeros((H,W-1,C),dtype=image.dtype)
    idx = np.array(range(len(seam)))*W + seam
    for i in range(C):
        out[:,:,i] = np.delete(image[:,:,i],(idx)).reshape((H,W-1))
        
    ### END YOUR CODE
    out = np.squeeze(out)  # remove last dimension if C == 1
   
    # Make sure that `out` has same type as `image`
    assert out.dtype == image.dtype, \
       "Type changed between image (%s) and out (%s) in remove_seam" % (image.dtype, out.dtype)

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process.

    At each step, we remove the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, 3)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, 3) if axis=0, or (H, size, 3) if axis=1
    """

    out = np.copy(image)
    
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))
    
    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    for i in range(W-size):
        energy = efunc(out)
        vcost, vpaths = cfunc(out,energy) 
        end = np.argmin(vcost[-1])
        seam_energy = vcost[-1, end]
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(out,seam)
        
    ### END YOUR CODE
    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def duplicate_seam(image, seam):
    """Duplicates pixels of the seam, making the pixels on the seam path "twice larger".
     
    This function will be helpful in functions enlarge_naive and enlarge.

    Args:
        image: numpy array of shape (H, W, C)
        seam: numpy array of shape (H,) of indices

    Returns:
        out: numpy array of shape (H, W+1, C)
    """

    H, W, C = image.shape
    out = np.zeros((H, W + 1, C))
    ### YOUR CODE HERE
    idx = np.array(range(len(seam)))*W + seam
    for i in range(C):
        mapCurr = image[:,:,i]
        #out[:,:,i] = np.insert(mapCurr,(idx),(mapCurr.reshape((H*W))[idx])).reshape((H,W+1))
        out[:,:,i] = np.insert(mapCurr,(idx),(mapCurr[np.arange(H),seam])).reshape((H,W+1))
    ### END YOUR CODE

    return out


def enlarge_naive(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    
    """Increases the size of the image using the seam duplication process.

    At each step, we duplicate the lowest energy seam from the image. We repeat the process
    until we obtain an output of desired size.
    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to increase height or width to (depending on axis)
        axis: increase in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert size > W, "size must be greather than %d" % W

    ### YOUR CODE HERE
    for i in range(size-W):
        energy = efunc(out)
        vcost, vpaths = cfunc(image,energy) 
        end = np.argmin(vcost[-1])
        seam_energy = vcost[-1, end]
        seam = backtrack_seam(vpaths, end)
        out = duplicate_seam(out,seam)
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def find_seams(image, k, axis=1, efunc=energy_function, cfunc=compute_cost):
    
    """Find the top k seams (with lowest energy) in the image.

    We act like if we remove k seams from the image iteratively, but we need to store their
    position to be able to duplicate them in function enlarge.

    We keep track of where the seams are in the original image with the array seams, which
    is the output of find_seams.
    We also keep an indices array to map current pixels to their original position in the image.

    Use functions:
        - efunc
        - cfunc
        - backtrack_seam
        - remove_seam

    Args:
        image: numpy array of shape (H, W, C)
        k: number of seams to find
        axis: find seams in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        seams: numpy array of shape (H, W)
    """

    image = np.copy(image)
    if axis == 0:
        image = np.transpose(image, (1, 0, 2))

    H, W, C = image.shape
    assert W > k, "k must be smaller than %d" % W

    # Create a map to remember original pixel indices
    # At each step, indices[row, col] will be the original column of current pixel
    # The position in the original image of this pixel is: (row, indices[row, col])
    # We initialize `indices` with an array like (for shape (2, 4)):
    #     [[1, 2, 3, 4],
    #      [1, 2, 3, 4]]
    indices = np.tile(range(W), (H, 1))  # shape (H, W)

    # We keep track here of the seams removed in our process
    # At the end of the process, seam number i will be stored as the path of value i+1 in `seams`
    # An example output for `seams` for two seams in a (3, 4) image can be:
    #    [[0, 1, 0, 2],
    #     [1, 0, 2, 0],
    #     [1, 0, 0, 2]]
    seams = np.zeros((H, W), dtype=np.int)

    # Iteratively find k seams for removal
    for i in range(k):
        # Get the current optimal seam
        energy = efunc(image)
        cost, paths = cfunc(image, energy)
        end = np.argmin(cost[H - 1])
        seam = backtrack_seam(paths, end)

        # Remove that seam from the image
        image = remove_seam(image, seam)

        # Store the new seam with value i+1 in the image
        # We can assert here that we are only writing on zeros (not overwriting existing seams)
        assert np.all(seams[np.arange(H), indices[np.arange(H), seam]] == 0), \
            "we are overwriting seams"
        seams[np.arange(H), indices[np.arange(H), seam]] = i + 1

        # We remove the indices used by the seam, so that `indices` keep the same shape as `image`
        indices = remove_seam(indices, seam)

    if axis == 0:
        seams = np.transpose(seams, (1, 0))

    return seams


def enlarge(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Enlarges the size of the image by duplicating the low energy seams.

    We start by getting the k seams to duplicate through function find_seams.
    We iterate through these seams and duplicate each one iteratively.

    Use functions:
        - find_seams
        - duplicate_seam

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: enlarge in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    # Transpose for height resizing
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H, W, C = out.shape

    assert size > W, "size must be greather than %d" % W

    assert size <= 2 * W, "size must be smaller than %d" % (2 * W)

    ### YOUR CODE HERE
    seams = find_seams(out, size-W)
 
    for i in range(size-W):
        # note that the size of "out" increases by 1 at each iteration
        # which we should consider when indexing using elements in "seams"
        out = duplicate_seam(out,(i-1)+seams[np.arange(H),np.where(seams == i+1)[1]])
       
    ### END YOUR CODE

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def compute_forward_cost(image, energy):
    
    """
    Computes forward cost map (vertical) and paths of the seams.

    Starting from the first row, compute the cost of each pixel as the sum of energy along the
    lowest energy path from the top.
    Make sure to add the forward cost introduced when we remove the pixel of the seam.

    We also return the paths, which will contain at each pixel either -1, 0 or 1 depending on
    where to go up if we follow a seam at this pixel.

    Args:
        image: numpy array of shape (H, W, 3) or (H, W)
        energy: numpy array of shape (H, W)

    Returns:
        cost: numpy array of shape (H, W)
        paths: numpy array of shape (H, W) containing values -1, 0 or 1
    """

    image = color.rgb2gray(image)
    H, W = image.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    # Initialization
    cost[0] = energy[0]
    for j in range(W):
        if j > 0 and j < W - 1:
            cost[0, j] += np.abs(image[0, j+1] - image[0, j-1])
    paths[0] = 0  # we don't care about the first row of paths

    ### YOUR CODE HERE
    # iterate over rows
    for r in range(H-1):
        r += 1
        # leave out the first row
        # dynamic programming throuogh vectorized codes
        # shift the last row of cost matrix to the left
        # and the right respectively to find the minimums 
        # every three elements
        lastRow = np.copy(cost[r-1,:])
        lastRowShiftedR = np.insert(lastRow[:-1],0,lastRow[0])
        lastRowShiftedL = np.insert(lastRow[1:],-1,lastRow[-1])
        '''
        lastRowShiftedR = np.roll(lastRow, 1)
        lastRowShiftedL = np.roll(lastRow, -1)
        lastRowShiftedR[0] = lastRow[0]
        lastRowShiftedL[-1] = lastRow[-1]
        '''
        # using a different manner to shift 1d array. just for a try
        
        # shift the last row in the image to both left and right directions.
        # vacant position is filled with head or tail elment.
        lastRowPix = np.copy(image[r-1,:])
        lastRowPixR = np.roll(lastRowPix,1)
        lastRowPixR[0] = lastRowPix[0]
        lastRowPixL = np.roll(lastRowPix,-1)
        lastRowPixL[-1] = lastRowPix[-1]
        # shift the current row in the image to both left and right directions.
        # vacant position is filled with 0.
        currRowPix = np.copy(image[r,:])
        currRowPixR = np.roll(currRowPix,1)
        currRowPixR[0] = currRowPix[0]
        currRowPixL = np.roll(currRowPix,-1)
        currRowPixL[-1] = currRowPix[-1]
        # compute differences in different directions
        diffRL = np.abs(currRowPixL - currRowPixR)
        diffRL[0]=0
        diffRL[-1]=0
        diffUL = np.abs(lastRowPix  - currRowPixR)
        diffUL[0] = 0
        diffUR = np.abs(lastRowPix  - currRowPixL)
        diffUR[-1] = 0
        diffULUR = np.abs(lastRowPixL - lastRowPixR)
        diffULUR[0] = 0
        diffULUR[-1] = 0
        # compute auxiliary cost CL, CR, and CV
        CL = diffRL + diffUL
        CR = diffRL + diffUR
        CV = diffRL
        #CV = diffRL + diffULUR
      
        # combine the three rows obtained above into a 2D array
        # and then use np.amin to find minimums in each column
        # find the indices of the mins as well
        
        mat4FindMins = np.stack((lastRowShiftedR+CL, lastRow+CV,lastRowShiftedL+CR))
        minVals = np.amin(mat4FindMins,axis=0)
        minIdces = np.argmin(mat4FindMins,axis=0)
        # add the current row with the newly found row of minimum costs,
        # and the result then replaces the current row
        
        cost[r,:] = energy[r,:] + minVals
        
        # use obtianed indices of mins to fill "paths"
        minIdces -= 1
        '''
        minIdces[minIdces == 0] = -1
        minIdces[minIdces == 1] = 0
        minIdces[minIdces == 2] = 1
        '''
        # shifting the row to the right and inserting a copy of the 
        # original row's first element into the array as the new head
        # may lead to a -1, which is incorrect and should be 0
        
        if minIdces[0] == -1:
            minIdces[0] = 0
       
        paths[r,:] = minIdces   
    ### END YOUR CODE

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def reduce_fast(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    """Reduces the size of the image using the seam carving process. Faster than `reduce`.

    Use your own implementation (you can use auxiliary functions if it helps like `energy_fast`)
    to implement a faster version of `reduce`.

    Args:
        image: numpy array of shape (H, W, C)
        size: size to reduce height or width to (depending on axis)
        axis: reduce in width (axis=1) or height (axis=0)
        efunc: energy function to use
        cfunc: cost function to use

    Returns:
        out: numpy array of shape (size, W, C) if axis=0, or (H, size, C) if axis=1
    """

    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    ### YOUR CODE HERE
    # Delete that line, just here for the autograder to pass setup checks
    out = reduce(image, size, 1, efunc, cfunc)
    pass
    ### END YOUR CODE

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out


def remove_object(image, mask):
    """Remove the object present in the mask.

    Returns an output image with same shape as the input image, but without the object in the mask.

    Args:
        image: numpy array of shape (H, W, 3)
        mask: numpy boolean array of shape (H, W)

    Returns:
        out: numpy array of shape (H, W, 3)
    """
    assert image.shape[:2] == mask.shape
     
    #H, W, _ = image.shape
    out = np.copy(image)
   
    out = np.transpose(out, (1, 0, 2))
    mask = np.transpose(mask)
    H, W, _ = out.shape
    ### YOUR CODE HERE
    mask = mask*1
    idx = np.where(mask == 1)
    W_new = W - (np.amax(idx[1])-np.amin(idx[1]))
    for i in range(W - W_new):
        energy = energy_function(out)
        energy[idx[0],idx[1]] = -999
        
        vcost, vpaths = compute_forward_cost(out,energy) 
        end = np.argmin(vcost[-1])
        seam_energy = vcost[-1, end]
        seam = backtrack_seam(vpaths, end)
        out = remove_seam(out,seam)
        mask = remove_seam(mask,seam)
        idx = np.where(mask == 1)
    
    print(out.shape)
    out = enlarge(out, W)
    ### END YOUR CODE
    out = np.transpose(out, (1, 0, 2))
    print(out.shape,image.shape)
    assert out.shape == image.shape

    return out
