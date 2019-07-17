"""
CS131 - Computer Vision: Foundations and Applications
Assignment 5
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/25/2018
Python Version: 3.5+
"""

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    centers = np.transpose(centers)
    assignments = np.zeros(N)
    
    
    
    featsNormalized = np.copy(features)
    if np.all(np.linalg.norm(features,axis = 1,keepdims=True)*1) == 0:
         featsNormalized = featsNormalized
    else:
         featsNormalized = features/np.linalg.norm(features,axis = 1,keepdims=True)
        #for n in range(num_iters):
    for n in range(num_iters):
        ### YOUR CODE HERE
        
        # using cosine similarity measure
        centersPre = np.copy(centers)
        if np.all(np.linalg.norm(centers,axis = 0,keepdims=True)*1) == 0:
            centers = centers
        else:
            centers = centers/np.linalg.norm(centers,axis = 0,keepdims=True)
        
        sims = np.dot(featsNormalized,centers)
        
        assignments = np.argmax(sims,axis=1)
       
        for i in range(k):
            cluster_i = features[assignments==i]
            centers[:,i] = np.mean(cluster_i,axis = 0)
        
        if np.all(centers == centersPre):
            print(n)
            break
        ### END YOUR CODE
        
    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    
    dists = np.zeros((N,k))
    
  
    for n in range(num_iters):
        
        centersPre = np.copy(centers)
        
        repFeatures = np.zeros((N,D,1))
        repFeatures[:,:,0] = np.copy(features)
        repFeatures = np.tile(repFeatures,(1,1,k))

        repCenters  = np.zeros((1,D,k))
        repCenters[0,:,:] = np.copy(np.transpose(centers))
        repCenters = np.tile(repCenters,(N,1,1))
        
        dists = np.linalg.norm(repFeatures - repCenters,axis=1)
        
        assignments = np.argmin(dists,axis = 1)
        
        for i in range(k):
            cluster_i = features[assignments==i]
            centers[i,:] = np.mean(cluster_i,axis = 0)
        
        if np.all(centers == centersPre):
            print(n)
            break
            
    return assignments



def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N)
    centers = np.copy(features)
    n_clusters = N
    
    # a list containing merged centers
    centerIdxs = []
    #oriCenterIdxs = []
    for i in range(N):
        centerIdxs.append([i])
        #oriCenterIdxs.append([i])
    
    # first initialized as long as "centers"
    # after one iteration, this becomes a 1D
    # array containing merged center(mean)
    mergedCenter = np.copy(centers) 
    
    
    # defining "dists" as a matrix when dealing with initialized
    # centroids. In a shceme where there are 400 feature vectors,
    # the "dists" is of size 400x400 with its diagonal elements being
    # all 0's, meaning the distance to selves is 0.

    # After the first merged centroid is found, the "dists" becomes
    # a matrix of size 399x399, since two original centers are merged
    # into one. At last, the size of this matrix become kxk
    
    dists = np.zeros((N,N))
    
    # calculating the differences in the x, y, and remaining directions
    # sequentially, taking the square of them and then adding them together.
    # This is done for calculating the Euclidean distance.
    for i in range(D):
        currCord = np.array([centers[:,i]])
        dists += (currCord - np.transpose(currCord))**2
    # Taking the square root of the result obtained in the last step
    # , obtaining the final Euclidean distance matrix.
    dists = np.sqrt(dists)
    
    # the index of the newly obtained merged center
    # first initialized as a sequence of N, then 
    # becomes a scalar
    newCenterIdx = np.arange(N)
    
   
    while n_clusters > k:
        ### YOUR CODE HERE

        # assigning very large values to diagonal elements to avoid
        # 0's being recognized as minimums.
        dists[newCenterIdx ,newCenterIdx ] = 999
        
        # First finding minimums in each row, and based on this finding
        # the indice of the final minimum.
        respectiveMatch = np.amin(dists,axis=1)
        # every element stands for an index of the closest point to
        # the one with the index as the position in this array
        #[3,58,4] ->(0,3) (1,58) (2,4)
        idxsRM = np.argmin(dists,axis = 1)
        # finding the closest pair among all pairs
        minPosRM = np.argmin(respectiveMatch)
         # [ [] , [] ]
        #mergedIdx = np.array([oriCenterIdxs[minPosRM],oriCenterIdxs[idxsRM[minPosRM]]])
        # the index pair of the closest one
        tempIdx =  np.array([minPosRM,idxsRM[minPosRM]])

        newCenterIdx = tempIdx[0]
        toBeRemovedIdx = tempIdx[1]
        
        #oriCenterIdxs.pop(toBeRemovedIdx)
        
        # merged indices
        centerIdxs[newCenterIdx] = list(centerIdxs[newCenterIdx][:])+ list(centerIdxs[toBeRemovedIdx][:])
        # merged centers
        mergedCenter = np.mean(features[centerIdxs[newCenterIdx],:],axis=0)
        # pop out one element
        centerIdxs.pop(toBeRemovedIdx)
        
        # update the array of centers
        centers = np.delete(centers, tempIdx, 0)
        centers = np.insert(centers,newCenterIdx,mergedCenter,0)
        
        # only calculating the distance between the merged center
        # and other centers. And then plugging it into "dists"
        distsMerged = np.zeros((1,centers.shape[0]))
        
        for i in range(D):
            currCord = np.array([centers[:,i]])
            currCordMerged = np.repeat(np.array([mergedCenter[i]], ndmin=2),centers.shape[0],axis = 1)
            distsMerged += (currCordMerged - currCord)**2
        distsMerged = np.sqrt(distsMerged)
        
        # deleting and plugging
        dists = np.delete(dists,toBeRemovedIdx,0)
        dists = np.delete(dists,toBeRemovedIdx,1)
        
        dists = np.insert(dists,newCenterIdx,distsMerged,0)
        dists = np.delete(dists,newCenterIdx+1,0)
        
        dists = np.insert(dists,newCenterIdx,distsMerged,1)
        dists = np.delete(dists,newCenterIdx+1,1)
        
        n_clusters = dists.shape[0]
       
        
        ### END YOUR CODE

    for i in range(k):
        assignments[centerIdxs[i]] = i

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))
    
    ### YOUR CODE HERE
    features = np.reshape(np.transpose(img,(2,0,1)),(C,-1))
    features = features.T
    ### END YOUR CODE

    return features

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    pos = np.transpose(np.mgrid[0:H,0:W],(1,2,0))
    color = np.dstack((color,pos))
    features = color_features(color)
    features = scale(features, -1, 1)
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """
    
    H,W = mask_gt.shape
    accuracy = None
    ### YOUR CODE HERE
    idx = mask_gt == mask
    accuracy = np.sum(idx*1)/(H*W)
   # print(accuracy)
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
