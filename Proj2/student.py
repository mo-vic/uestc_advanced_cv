import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def get_interest_points(image, feature_width):
    '''
    Returns a set of interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here!

    # Compute $I_x^2$, $I_y^2$, $I_{xy}$, $I_{yx}$
    partial_x = filters.sobel_h(image)
    partial_x2 = filters.sobel_h(partial_x)
    partial_y = filters.sobel_v(image)
    partial_y2 = filters.sobel_v(partial_y)
    partial_xy = filters.sobel_v(partial_x)
    partial_yx = filters.sobel_h(partial_y)


    # Apply Gaussian filter, equivalent to compute elements of $M$
    sigma = 1.0
    partial_x2 = filters.gaussian(partial_x2, sigma=sigma)
    partial_y2 = filters.gaussian(partial_y2, sigma=sigma)
    partial_xy = filters.gaussian(partial_xy, sigma=sigma)
    partial_yx = filters.gaussian(partial_yx, sigma=sigma)

    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    # Construct matrix M in all location
    M = np.zeros(h * w * 4, np.float32)
    M[0::4] = partial_x2.flatten()
    M[1::4] = partial_xy.flatten()
    M[2::4] = partial_yx.flatten()
    M[3::4] = partial_y2.flatten()
    M = M.reshape((-1, 2, 2))

    det_M = np.linalg.det(M)
    tr_M = np.trace(M, axis1=1, axis2=2)

    # Parameters used in harris corner detector
    k = 0.04
    c = 0.02

    # Compute cornerness score
    cornerness = det_M - k * np.square(tr_M)

    # For $|R| < \epsilon$, used to remove flat area
    cornerness[cornerness < c * cornerness.max()] = 0

    # Recover shape
    cornerness = cornerness.reshape((h, w))

    # Non max suppression
    coordinates = feature.peak_local_max(cornerness, min_distance=15)

    # Be aware of the index for xs and ys
    xs = coordinates[:, 1]
    ys = coordinates[:, 0]

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
    Returns a set of feature descriptors for a given set of interest points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here!

    if len(image.shape) == 2:
        h, w = image.shape
    else:
        h, w, _ = image.shape

    offset = feature_width // 2
    num_points = x.shape[0]

    # Range of index
    x_start = x - offset
    x_stop = x + offset
    y_start = y - offset
    y_stop = y + offset

    # Compute padding parameters
    x_min = x_start.min()
    x_max = x_stop.max()
    y_min = y_start.min()
    y_max = y_stop.max()

    x_pad = [0, 0]
    y_pad = [0, 0]
    if x_min < 0:
        x_pad[0] = -x_min
    if y_min < 0:
        y_pad[0] = -y_min
    if x_max - w >= 0:
        x_pad[1] = x_max - w + 1
    if y_max - h >= 0:
        y_pad[1] = y_max - h + 1

    x_start += x_pad[0]
    x_stop += x_pad[0]
    y_start += y_pad[0]
    y_stop += y_pad[0]

    # Pad the input image
    image = np.pad(image, [y_pad, x_pad], mode="constant")

    # Index under the window for each dimension
    cell_size = 4
    num_blocks = feature_width // cell_size

    x_idx = np.array([np.arange(start, stop) for start, stop in zip(x_start, x_stop)])
    y_idx = np.array([np.arange(start, stop) for start, stop in zip(y_start, y_stop)])

    # Before transpose -> (num_blocks, num_points, cell_size)
    x_idx = np.array(np.split(x_idx, num_blocks, axis=1))
    y_idx = np.array(np.split(y_idx, num_blocks, axis=1))

    # After transpose -> (num_points, num_blocks, cell_size)
    x_idx = x_idx.transpose([1, 0, 2])
    y_idx = y_idx.transpose([1, 0, 2])

    # Index for every pixel under the window
    x_idx = np.tile(np.tile(x_idx, cell_size), [1, num_blocks, 1]).flatten()
    y_idx = np.tile(np.repeat(y_idx, cell_size, axis=2), num_blocks).flatten()

    # Compute partial derivatives
    partial_x = filters.sobel_h(image)
    partial_y = filters.sobel_v(image)
    # Comput gradient magnitude
    magnitude = np.sqrt(partial_x * partial_x + partial_y * partial_y)
    # Compute gradient orientation
    orientation = np.arctan2(partial_y, partial_x) + np.pi
    # Assign gradient to the nearest angle. Should use round instead of floor
    orientation = np.mod(np.round(orientation / (2.0 * np.pi) * 8.0), 8)
    orientation = orientation.astype(np.int32)
    # Smooth the gradient magnitude
    magnitude = filters.gaussian(magnitude, sigma=offset)

    # Retrieve values of all patches as an 1D array
    magnitude_in_pixels = magnitude[y_idx, x_idx]
    orientation_in_pixels = orientation[y_idx, x_idx]

    # Reshape the pixel array to (num_patches, cell_size, cell_size)
    magnitude_in_cells = magnitude_in_pixels.reshape((-1, cell_size * cell_size))
    orientation_in_cells = orientation_in_pixels.reshape((-1, cell_size * cell_size))

    # Compute weight sum of orientations in each cell
    features = np.array(list(map(lambda array, weight : np.bincount(array, weight, minlength=8), orientation_in_cells, magnitude_in_cells)))

    # Reshape the features to (num_points, feature_length), each row represents the feature for a keypoint
    # Normalize -> Clamp -> Renormalize
    features = features.reshape((num_points, -1))
    features = features / np.linalg.norm(features, axis=-1).reshape((-1, 1))
    features[features >= 0.2] = 0.2
    features = features / np.linalg.norm(features, axis=-1).reshape((-1, 1))

    return features


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - zip (python built in function)

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here!

    m, n = im1_features.shape[0], im2_features.shape[0]
    dist = np.repeat(im1_features, n, axis=0) - np.tile(im2_features, [m, 1])
    dist = np.sqrt(np.sum(np.square(dist), axis=1))
    dist = dist.reshape((m, n))
    sorted_index = np.argsort(dist, axis=1)

    thres = 0.95
    aux_idx = np.arange(m)
    ratio = dist[aux_idx, sorted_index[:, 0]] / dist[aux_idx, sorted_index[:, 1]]
    ratio[np.isnan(ratio)] = 1.0

    matched_idx = (ratio <= thres)
    confidences = 1.0 - ratio[ratio <= thres]
    matches = np.stack([aux_idx[matched_idx], sorted_index[matched_idx, 0]], axis=1)

    return matches, confidences
