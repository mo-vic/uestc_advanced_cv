from utils import *

import os.path as osp
from glob import glob
from functools import partial

import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from skimage.transform import rescale


def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.

    Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)

        You can also use cyvlfeat, a vlfeat lib for python. 

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """

    # Params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    orientations = feature_params.get('orientations', 31)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = win_size // cell_size

    """
    In practice, cells_per_block = (1, 1) works better than cell_per_block = (n_cell, n_cell). However, setting 
    cells_per_block = (1, 1) makes detection or mining hard example a bit cumbersome. Make sure to re-arrange 
    the HoG features that matches the structure of the template if you set cells_per_block = (1, 1). And I found 
    cells_per_block = (1, 1) works better only in terms of training accuracy, the testing performance is terrible.
    """
    f = lambda x: hog(load_image_gray(x), orientations=orientations, pixels_per_cell=(cell_size, cell_size),
                      cells_per_block=(1, 1), feature_vector=True).reshape((-1, n_cell ** 2 * orientations))
    feats = np.concatenate(list(map(f, positive_files)))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats


def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).

    Useful functions:
    -   Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)
    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """

    # Params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    orientations = feature_params.get('orientations', 31)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = win_size // cell_size
    # Number of patches to be sampled from each image
    num_per_img = num_samples // len(negative_files)
    # Compute the rest number of patches
    residual = num_samples % len(negative_files)

    num_per_img = np.full(len(negative_files), num_per_img)
    # Randomly dispatch the residual
    num_per_img[np.random.permutation(len(negative_files))[:residual]] += 1

    """
    In practice, cells_per_block = (1, 1) works better than cell_per_block = (n_cell, n_cell). However, setting 
    cells_per_block = (1, 1) makes detection or mining hard example a bit cumbersome. Make sure to re-arrange 
    the HoG features that matches the structure of the template if you set cells_per_block = (1, 1). And I found 
    cells_per_block = (1, 1) works better only in terms of training accuracy, the testing performance is terrible.
    """
    f = lambda x: (lambda im=load_image_gray(x[0]): [
        hog(im[start:start + win_size, stop:stop + win_size], orientations=orientations,
            pixels_per_cell=(cell_size, cell_size), cells_per_block=(1, 1), feature_vector=True).reshape(
            (-1, n_cell ** 2 * orientations))
        for start, stop in
        zip(np.random.randint(0, im.shape[0] - win_size, x[1]), np.random.randint(0, im.shape[1] - win_size, x[1]))])()

    feats = np.concatenate(list(map(f, zip(negative_files, num_per_img)))).squeeze(1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats


def train_classifier(features_pos, features_neg, C, class_weight={1: 10}):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    N, D = features_pos.shape
    M, D = features_neg.shape

    train_feats = np.vstack([features_pos, features_neg])
    train_labels = np.hstack([np.ones(N), -np.ones(M)])

    svm = LinearSVC(max_iter=10000, C=C, class_weight=class_weight)
    svm.fit(train_feats, train_labels)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm


def slide2d(array, win_size, stride=1):
    """
    2D sliding window operation.
    Used to convert input array to patches.
    """

    m, n, n_cells_row, n_cells_col, n_orient = array.shape

    # Works for stride == 1 for now
    assert stride == 1

    # Number of blocks in row
    n_blocks_row = n - win_size + 1
    # Number of blocks in column
    n_blocks_col = m - win_size + 1

    # Index in each dimension
    i = np.repeat(np.arange(0, win_size).reshape((1, -1)), n_blocks_col, axis=0)
    j = np.repeat(np.arange(0, win_size).reshape((1, -1)), n_blocks_row, axis=0)
    i = i + np.arange(0, n_blocks_col).reshape((n_blocks_col, 1))
    j = j + np.arange(0, n_blocks_row).reshape((n_blocks_row, 1))

    # Broadcasting
    j = np.tile(np.tile(j, win_size), [n_blocks_col, 1]).flatten()
    i = np.tile(np.repeat(i, win_size, axis=1), n_blocks_row).flatten()

    # Retrieve the element
    blocks = np.array(array[i, j])

    blocks = blocks.reshape((n_blocks_col, n_blocks_row, win_size * win_size, n_cells_row, n_cells_col, n_orient))
    # print("Shape of blocks", blocks.shape)

    # # Identity check
    # check = []
    # for i in range(0, n_blocks_col):
    #     for j in range(0, n_blocks_row):
    #         check.append(np.all(array[i:i+win_size, j:j+win_size].reshape((win_size*win_size, n_cells_row, n_cells_col, n_orient)) == blocks[i, j]))
    # print("All check:", np.all(check))

    return blocks


def mining(file_path, svm, win_size, cell_size, orientations):
    # Loading image
    image = load_image_gray(file_path)

    # Extract HoG features
    hog_feats = hog(image, orientations=orientations, pixels_per_cell=(cell_size, cell_size), cells_per_block=(1, 1),
                    feature_vector=False)

    # Group HoG features into blocks by sliding over `hog_feats`
    n_cell = win_size // cell_size
    hog_feats = slide2d(hog_feats, win_size=n_cell)

    # Reshape to (`num_blocks`, -1), here `num_blocks` matches win_size, i.e., the structure of the template
    n_blocks_col, n_blocks_row = hog_feats.shape[:2]
    hog_feats = hog_feats.reshape((n_blocks_col * n_blocks_row, -1))

    # Make predictions on all blocks
    predictions = svm.predict(hog_feats)
    # Pick the HoG features that predicted as false positive to be our hard negative examples
    false_positives = hog_feats[predictions == 1]

    # # Compute confidence for all blocks
    # confidences = svm.decision_function(hog_feats)
    # # Pick the HoG features that predicted as false positive to be our hard negative examples
    # false_positives = hog_feats[np.logical_and(0 <= confidences, confidences <= 0.01)]

    return false_positives


def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # Params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    orientations = feature_params.get('orientations', 31)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    feats = np.concatenate(list(
        map(partial(mining, svm=svm, win_size=win_size, cell_size=cell_size, orientations=orientations),
            negative_files)))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return feats


def run_detector(test_scn_path, svm, feature_params, threshold=0.5, verbose=False):
    """
    This function returns detections on all of the images in a given path. You
    will want to use non-maximum suppression on your detections or your
    performance will be poor (the evaluation counts a duplicate detection as
    wrong). The non-maximum suppression is done on a per-image basis. The
    starter code includes a call to a provided non-max suppression function.

    The placeholder version of this code will return random bounding boxes in
    each test image. It will even do non-maximum suppression on the random
    bounding boxes to give you an example of how to call the function.

    Your actual code should convert each test image to HoG feature space with
    a _single_ call to hog() for each scale. Then step over the HoG
    cells, taking groups of cells that are the same size as your learned
    template, and classifying them. If the classification is above some
    confidence, keep the detection and then pass all the detections for an
    image to non-maximum suppression. For your initial debugging, you can
    operate only at a single scale and you can skip calling non-maximum
    suppression. Err on the side of having a low confidence threshold (even
    less than zero) to achieve high enough recall.
    
    Useful functions:
    -   skimage.feature.hog(im, pixels_per_cell = (*, *)): computes HoG features
        eg:
                skimage.feature.hog(im, pixels_per_cell=(cell_size, cell_size),
                         cells_per_block=(n_cell, n_cell),
                         orientations=31)

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """

    bboxes = []
    image_ids = []
    confidences = []
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))

    # Number of top detections to feed to NMS
    topk = 200

    # Confidence threshold
    thres = threshold

    # Params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
    orientations = feature_params.get('orientations', 31)

    n_cell = win_size // cell_size
    multi_scale_factor = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape

        bboxes_per_image = []
        confidences_per_image = []

        #######################################################################
        #                        TODO: YOUR CODE HERE                         #
        #######################################################################
        # 1. create scale space HOG pyramid and return scores for prediction

        # 2. scale image. We suggest you create a scale factor list and use recurrence 
        #    to generate image feature. eg:
        #       multi_scale_factor = np.array([1.0, 0.7, 0.5, 0.3])
        #       for scala_rate in multi_scale_factor:
        #           scale img
        #           xxx
        # 3. image to hog feature
        # 4. sliding windows at scaled feature map. you can use horizontally 
        #    recurrence and vertically recurrence
        # 5. extract feature for current bounding box and use classify
        # 6. record your result for this image
        # 7. non-maximum suppression (NMS)
        #    non_max_supr_bbox() can actually get somewhat slow with thousands of
        #    initial detections. You could pre-filter the detections by confidence,
        #    e.g. a detection with confidence -1.1 will probably never be
        #    meaningful. You probably _don't_ want to threshold at 0.0, though. You
        #    can get higher recall with a lower threshold. You should not modify
        #    anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        #    please create another function.
        #    eg:
        #     is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
        #        im_shape, verbose=verbose)

        for scale_rate in multi_scale_factor:
            scaled_im = rescale(im, scale=scale_rate, anti_aliasing=True)
            hog_feats = hog(scaled_im, orientations=orientations, pixels_per_cell=(cell_size, cell_size),
                            cells_per_block=(1, 1), feature_vector=False)

            n_blocks_col, n_blocks_row = hog_feats.shape[:2]
            if (n_blocks_col // n_cell) * (n_blocks_row // n_cell) == 0:
                break

            hog_feats = slide2d(hog_feats, win_size=n_cell)

            # Reshape to (`num_blocks`, -1), here `num_blocks` matches win_size, i.e., the structure of the template
            n_blocks_col, n_blocks_row = hog_feats.shape[:2]
            hog_feats = hog_feats.reshape((n_blocks_col * n_blocks_row, -1))

            # Compute confidence on all blocks
            confidences_per_scale = svm.decision_function(hog_feats)

            # Reshape confidences to (n_blocks_col, n_blocks_row)
            confidences_per_scale = confidences_per_scale.reshape((n_blocks_col, n_blocks_row))

            detected_faces = (confidences_per_scale >= thres)
            confidences_per_scale = confidences_per_scale[detected_faces].flatten()

            if confidences_per_scale.size == 0:
                continue

            (block_ys, block_xs) = np.where(detected_faces)
            x_min = block_xs * cell_size
            y_min = block_ys * cell_size
            x_max = block_xs * cell_size + win_size
            y_max = block_ys * cell_size + win_size

            bboxes_per_scale = np.array([x_min, y_min, x_max, y_max]).T
            bboxes_per_scale = bboxes_per_scale / scale_rate
            bboxes_per_image.append(bboxes_per_scale)
            confidences_per_image.append(confidences_per_scale)

        if len(bboxes_per_image) == 0:
            continue

        bboxes_per_image = np.concatenate(bboxes_per_image, axis=0).astype(np.int32)
        confidences_per_image = np.hstack(confidences_per_image)

        sorted_index = np.argsort(confidences_per_image)[::-1]
        bboxes_per_image = bboxes_per_image[sorted_index[:topk]]
        confidences_per_image = confidences_per_image[sorted_index[:topk]]
        survives = non_max_suppression_bbox(bboxes_per_image, confidences_per_image, im_shape, verbose=verbose)

        bboxes_per_image = bboxes_per_image[survives]
        confidences_per_image = confidences_per_image[survives]

        bboxes.append(bboxes_per_image)
        confidences.append(confidences_per_image)
        image_ids.extend([im_id] * np.sum(survives))

    bboxes = np.concatenate(bboxes, axis=0)
    confidences = np.hstack(confidences)
    image_ids = np.array(image_ids)

    #######################################################################
    #                          END OF YOUR CODE                           #
    #######################################################################

    return bboxes, confidences, image_ids
