"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    dx2 = dx**2
    dy2 = dy**2
    dxdy = dx*dy
    dx2_conv = convolve(dx2,window)
    dy2_conv = convolve(dy2,window)
    dxdy_conv = convolve(dxdy,window)
    
    for y in range(H):
        for x in range(W):
            M = np.array([
                           [dx2_conv[y,x],dxdy_conv[y,x]],
                           [dxdy_conv[y,x],dy2_conv[y,x]]
                         ])
            response[y,x] = np.linalg.det(M)-k*(np.trace(M)**2)
    ### END YOUR CODE

    return response


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    if denom == 0:
        denom = 1
    return x_min + nom/denom 

def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.  

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    H,W = patch.shape
    feature = []
    ### YOUR CODE HERE
    patch = patch.reshape((H*W))
    feature = scale(patch, -0.5, 0.5)
    #print(feature.shape)
 
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
        
    
    return np.asarray(desc)


def match_descriptors(desc1, desc2, threshold):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    M = desc1.shape[0]
    #N = desc2.shape[0]
    #print(M,N)
    dists = cdist(desc1, desc2)
    #print(dists.shape)
    ### YOUR CODE HERE
    for row in range(M):
        idxMin = np.argmin(dists[row,:])
        distCurrentRowSorted = np.sort(dists[row,:])
        if row == 0:
            print(distCurrentRowSorted[0],distCurrentRowSorted[1])
        if distCurrentRowSorted[0]/distCurrentRowSorted[1]<=threshold:
            matches.append([row,idxMin])
    matches = np.asarray(matches)
    
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)

    Return:
        H: a matrix of shape (P, P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
   
    p1 = pad(p1)
    p2 = pad(p2)
  
    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2,p1)
    H = H[0]

    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()
    
    N = matches.shape[0]
    print(N)
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])
    
    max_inliers = np.zeros(N)
    n_inliers = 0
   
    
    # RANSAC iteration start
    ### YOUR CODE HERE
    HArray = np.zeros((3,3,n_iters))
    idxInliers = []
    numInliers = np.zeros(n_iters)
    for i in range(n_iters):
        idx = np.random.permutation(N)
        idx = idx[:n_samples]
        
        p1 = matched1[idx,:]
        p2 = matched2[idx,:]
       
        H = np.linalg.lstsq(p2,p1)
        H = H[0]
        H[:,2] = np.array([0, 0, 1])
        HArray[:,:,i] = H
        
        mat1Fitted = np.dot(matched2,H)
        
        dists = np.sqrt( (matched1[:,0]-mat1Fitted[:,0])**2 +
                         (matched1[:,1]-mat1Fitted[:,1])**2
                       )
     
        idxInliers.append(np.where(dists<threshold)[0])
        #print(idxInliers[i][0])
        numInliers[i] = idxInliers[i].shape[0]
    
    idxMaxInliers = np.argmax(numInliers)
    #print(numInliers)
    #print(idxMaxInliers)
    #print(numInliers[idxMaxInliers])
    max_inliers = idxInliers[idxMaxInliers]
   
   # print(idxInliers)
   # print(robustMats)
    H = HArray[:,:,idxMaxInliers]
        
    ### END YOUR CODE
    print(H)
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients 
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]
    
    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    for y in range(rows):
        for x in range(cols):
            curGCell = G_cells[y,x]
            curThetaCell = theta_cells[y,x]
            for BIN in range(n_bins):
                idxCurBin = np.where( (curThetaCell>BIN*degrees_per_bin) & 
                                      (curThetaCell<=(BIN+1)*degrees_per_bin))
                GCurBin = curGCell[idxCurBin[0],idxCurBin[1]]
                cells[y,x,BIN] = np.sum(GCurBin)
                #print("BIN, ",BIN)
                #print(cells[y,x,BIN])
            
    block = cells.reshape((rows*cols*n_bins))
    block = block/np.linalg.norm(block)
    #print(block.shape)
    
    ### YOUR CODE HERE

    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image
    
    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]
    
    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]

    ### YOUR CODE HERE
    wMat1 = np.ones(img1_mask.shape)
    wMat2 = np.ones(img2_mask.shape)
    
    wDecrease = np.tile(np.linspace(1,0,right_margin-left_margin+1),(out_H,1))
    wIncrease = np.tile(np.linspace(0,1,right_margin-left_margin+1),(out_H,1))
    
    print(wDecrease.shape,wIncrease.shape)
    print(wMat1[:,left_margin:right_margin+1].shape,wMat2[:,left_margin:right_margin+1].shape)
    
    
    wMat1[:,left_margin:right_margin+1] = wDecrease
    wMat2[:,left_margin:right_margin+1] = wIncrease
    
    merged = img1_warped*wMat1 + img2_warped*wMat2
    
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    HList = []
    for i in range(len(imgs)-1):
        H, robust_matches = ransac(keypoints[i], keypoints[i+1], matches[i], threshold=1)
        HList.append(H)
        '''
        p1 = keypoints[i][matches[i][:,0]]
        p2 = keypoints[i+1][matches[i][:,1]]
        H = fit_affine_matrix(p1, p2)
        print(H)
        HList.append(H)
        '''
    imgWarpedList = []
    idxRef = len(imgs)//2-1
    imgRef = imgs[idxRef]
    HMerged = []
    
    for i in range(len(imgs)):
        
        if i < idxRef:
            
            HRev = np.eye(3)
            for j in range(idxRef-i):
                HRev = np.dot(HRev,np.linalg.inv(HList[i+j]))
            
            HMerged.append(HRev)
            
            output_shape, offset = get_output_space(imgRef, [imgs[i]], [HRev])
            
            imgRef = warp_image(imgRef, np.eye(3), output_shape, offset)
            
            
            #imgWarpedRev = warp_image(img[i], HRev, output_shape, offset)
            #imgWarpedList.append(imgWarpedRev)
          
        elif i > idxRef:
            
            H = np.eye(3)
            for j in range(i-idxRef):
                H = np.dot(H,HList[i-j-1])
            HMerged.append(H)
            
            output_shape, offset = get_output_space(imgRef, [imgs[i]], [H])
            
            imgRef = warp_image(imgRef, np.eye(3), output_shape, offset)
            
            #imgWarpedRev = warp_image(img[i], HRev, output_shape, offset)
            #imgWarpedList.append(imgWarpedRev)
    imgMaskRef = (imgRef != -1) # Mask == 1 inside the image
    imgRef[~imgMaskRef] = 0     # Return background values to 0
    
    imgs.pop(idxRef)       
    for i in range(len(imgs)):
        imgWarped = warp_image(imgs[i], HMerged[i], output_shape, offset)
        imgMask = (imgWarped != -1) # Mask == 1 inside the image
        imgWarped[~imgMask] = 0     # Return background values to 0
        imgWarpedList.append(imgWarped)
     
    #panorama = imgWarpedList[0]+imgWarpedList[1]+imgWarpedList[2]+imgRef
        
    imgWarpedList.insert(idxRef,imgRef)
    merged = imgWarpedList[0]
    for i in range(len(imgWarpedList)-1):
            merged = linear_blend(merged, imgWarpedList[i+1])
            
    panorama = merged
     
    
    ### END YOUR CODE

    return panorama
