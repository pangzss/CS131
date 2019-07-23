"""
CS131 - Computer Vision: Foundations and Applications
Assignment 8
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 12/02/2017
Python Version: 3.5+
"""

import numpy as np
from skimage.transform import pyramid_gaussian
from skimage.filters import sobel_h, sobel_v, gaussian
from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage.filters import convolve

def lucas_kanade(img1, img2, keypoints, window_size=5):
    """ Estimate flow vector at each keypoint using Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).

    Hints:
        - You may use np.linalg.inv to compute inverse matrix
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    flow_vectors = []
    w = window_size // 2

    # Compute partial derivatives
    Iy, Ix = np.gradient(img1)
    It = img2 - img1
    
    # For each [y, x] in keypoints, estimate flow vector [vy, vx]
    # using Lucas-Kanade method and append it to flow_vectors.
    for y, x in keypoints:
        # Keypoints can be loacated between integer pixels (subpixel locations).
        # For simplicity, we round the keypoint coordinates to nearest integer.
        # In order to achieve more accurate results, image brightness at subpixel
        # locations can be computed using bilinear interpolation.
        y, x = int(round(y)), int(round(x))

        ### YOUR CODE HERE
        pat_Iy = Iy[y-w:y+w+1,x-w:x+w+1].reshape((window_size**2,1))
        pat_Ix = Ix[y-w:y+w+1,x-w:x+w+1].reshape((window_size**2,1))
        
        A = np.hstack((pat_Ix,pat_Iy))
        
        b = It[y-w:y+w+1,x-w:x+w+1].reshape((window_size**2,1))
        
        v = np.linalg.multi_dot([np.linalg.inv(np.dot(A.T,A)), A.T, b]).flatten()

        flow_vectors.append(v[:])
        
        ### END YOUR CODE
   
    flow_vectors = np.array(flow_vectors)
    
    return flow_vectors

def iterative_lucas_kanade(img1, img2, keypoints,
                           window_size=9,
                           num_iters=7,
                           g=None):
    """ Estimate flow vector at each keypoint using iterative Lucas-Kanade method.

    Args:
        img1 - Grayscale image of the current frame. Flow vectors are computed
            with respect to this frame.
        img2 - Grayscale image of the next frame.
        keypoints - Keypoints to track. Numpy array of shape (N, 2).
        window_size - Window size to determine the neighborhood of each keypoint.
            A window is centered around the current keypoint location.
            You may assume that window_size is always an odd number.
        num_iters - Number of iterations to update flow vector.
        g - Flow vector guessed from previous pyramid level.
    Returns:
        flow_vectors - Estimated flow vectors for keypoints. flow_vectors[i] is
            the flow vector for keypoint[i]. Numpy array of shape (N, 2).
    """
    assert window_size % 2 == 1, "window_size must be an odd number"

    # Initialize g as zero vector if not provided
    if g is None:
        g = np.zeros(keypoints.shape)

    flow_vectors = []
    w = window_size // 2
    window = np.ones((window_size, window_size))

    # Compute spatial gradients
    dy, dx = np.gradient(img1)
    
    dx2 = dx**2
    dy2 = dy**2
    dxdy = dx*dy
    dx2_conv = convolve(dx2,window)
    dy2_conv = convolve(dy2,window)
    dxdy_conv = convolve(dxdy,window)
    
    for y, x, gy, gx in np.hstack((keypoints, g)):
        v = np.zeros(2) # Initialize flow vector as zero vector
        y1 = int(round(y)); x1 = int(round(x))


        # TODO: Compute inverse of G at point (x1, y1)
        ### YOUR CODE HERE
        G = np.array([
                           [dx2_conv[y1,x1],dxdy_conv[y1,x1]],
                           [dxdy_conv[y1,x1],dy2_conv[y1,x1]]
                         ])
        G_inv = np.linalg.inv(G)
        ### END YOUR CODE

        # iteratively update flow vector
        for k in range(num_iters):
            vx, vy = v
            # Refined position of the point in the next frame
            y2 = int(round(y+gy+vy)); x2 = int(round(x+gx+vx))

            # TODO: Compute bk and vk = inv(G) x bk
            ### YOUR CODE HERE
           # print(img2.shape,y2-w,y2+w+1,x2-w,x2+w+1)
            H,W = img2.shape
            if y2-w<0 or y2+w+1>H or x2-w<0 or x2+w+1>W:
                continue
            delta_dk = img1[y1-w:y1+w+1,x1-w:x1+w+1] - img2[y2-w:y2+w+1,x2-w:x2+w+1]
            pat_dx   = dx[y1-w:y1+w+1,x1-w:x1+w+1]
            pat_dy   = dy[y1-w:y1+w+1,x1-w:x1+w+1]
            bk1      = np.sum(delta_dk * pat_dx)
            bk2      = np.sum(delta_dk * pat_dy)
            bk       = np.array([ [bk1],
                                  [bk2] ])
            vk = np.dot(G_inv,bk)
            #print(vk.shape)
            ### END YOUR CODE

            # Update flow vector by vk
            v += vk.flatten()

        vx, vy = v
        flow_vectors.append([vy, vx])

    return np.array(flow_vectors)
        

def pyramid_lucas_kanade(img1, img2, keypoints,
                         window_size=9, num_iters=7,
                         level=2, scale=2):

    """ Pyramidal Lucas Kanade method

    Args:
        img1 - same as lucas_kanade
        img2 - same as lucas_kanade
        keypoints - same as lucas_kanade
        window_size - same as lucas_kanade
        num_iters - number of iterations to run iterative LK method
        level - Max level in image pyramid. Original image is at level 0 of
            the pyramid.
        scale - scaling factor of image pyramid.

    Returns:
        d - final flow vectors
    """

    # Build image pyramids of img1 and img2
    pyramid1 = tuple(pyramid_gaussian(img1, max_layer=level, downscale=scale))
    pyramid2 = tuple(pyramid_gaussian(img2, max_layer=level, downscale=scale))

    # Initialize pyramidal guess
    g = np.zeros(keypoints.shape)

    for L in range(level, -1, -1):
        ### YOUR CODE HERE
        pos_L = keypoints/(scale**L)
        #H,W = pyramid1[L].shape
        #idx = (pos_L[:,0]>H) + (pos_L[:,1]>W)
        #idx = np.nonzero(idx)
        #pos_L = np.delete(pos_L,idx,axis=0).reshape((-1,2))
        d   = iterative_lucas_kanade(pyramid1[L], pyramid2[L], pos_L,
                           window_size,
                           num_iters,
                           g)
        if L > 0:
            g = scale*(g+d)
        ### END YOUR CODE

    d = g + d
    return d

def compute_error(patch1, patch2):
    """ Compute MSE between patch1 and patch2

        - Normalize patch1 and patch2
        - Compute mean square error between patch1 and patch2

    Args:
        patch1 - Grayscale image patch of shape (patch_size, patch_size)
        patch2 - Grayscale image patch of shape (patch_size, patch_size)
    Returns:
        error - Number representing mismatch between patch1 and patch2
    """
    assert patch1.shape == patch2.shape, 'Differnt patch shapes'
    error = 0
    ### YOUR CODE HERE
    patch1 = (patch1-patch1.mean())/patch1.std()
    patch2 = (patch2-patch2.mean())/patch2.std()
    
    error = ((patch1 - patch2)**2).mean()
    
    ### END YOUR CODE
    return error

def track_features(frames, keypoints,
                   error_thresh=1.5,
                   optflow_fn=pyramid_lucas_kanade,
                   exclude_border=5,
                   **kwargs):

    """ Track keypoints over multiple frames

    Args:
        frames - List of grayscale images with the same shape.
        keypoints - Keypoints in frames[0] to start tracking. Numpy array of
            shape (N, 2).
        error_thresh - Threshold to determine lost tracks.
        optflow_fn(img1, img2, keypoints, **kwargs) - Optical flow function.
        kwargs - keyword arguments for optflow_fn.

    Returns:
        trajs - A list containing tracked keypoints in each frame. trajs[i]
            is a numpy array of keypoints in frames[i]. The shape of trajs[i]
            is (Ni, 2), where Ni is number of tracked points in frames[i].
    """

    kp_curr = keypoints
    trajs = [kp_curr]
    patch_size = 3 # Take 3x3 patches to compute error
    w = patch_size // 2 # patch_size//2 around a pixel

    for i in range(len(frames) - 1):
        I = frames[i]
        J = frames[i+1]
        flow_vectors = optflow_fn(I, J, kp_curr, **kwargs)
        kp_next = kp_curr + flow_vectors

        new_keypoints = []
        for yi, xi, yj, xj in np.hstack((kp_curr, kp_next)):
            # Declare a keypoint to be 'lost' IF:
            # 1. the keypoint falls outside the image J
            # 2. the error between points in I and J is larger than threshold

            yi = int(round(yi)); xi = int(round(xi))
            yj = int(round(yj)); xj = int(round(xj))
            # Point falls outside the image
            if yj > J.shape[0]-exclude_border-1 or yj < exclude_border or\
               xj > J.shape[1]-exclude_border-1 or xj < exclude_border:
                continue

            # Compute error between patches in image I and J
            patchI = I[yi-w:yi+w+1, xi-w:xi+w+1]
            patchJ = J[yj-w:yj+w+1, xj-w:xj+w+1]
            error = compute_error(patchI, patchJ)
            if error > error_thresh:
                continue

            new_keypoints.append([yj, xj])

        kp_curr = np.array(new_keypoints)
        trajs.append(kp_curr)

    return trajs


def IoU(bbox1, bbox2):
    """ Compute IoU of two bounding boxes

    Args:
        bbox1 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
        bbox2 - 4-tuple (x, y, w, h) where (x, y) is the top left corner of
            the bounding box, and (w, h) are width and height of the box.
    Returns:
        score - IoU score
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    score = 0

    ### YOUR CODE HERE
    x1_arr = np.arange(w1)+x1
    y1_arr = np.arange(h1)+y1
    x2_arr = np.arange(w2)+x2
    y2_arr = np.arange(h2)+y2
    
    set1 = set()
    set2 = set()
    for i in range(h1):
        for j in range(w1):
            set1.add((y1_arr[i],x1_arr[j]))
    
    for i in range(h2):
        for j in range(w2):
            set2.add((y2_arr[i],x2_arr[j]))
            
    score = len(set1.intersection(set2))/len(set1.union(set2))
    ### END YOUR CODE

    return score


