# CSC320 Winter 2017
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
from math import log

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# Global variables for modularity
up,down,left,right = "u","d","l","r"

# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################



    #############################################

    return new_f, best_D, global_vars

# Helper that implements propagation
# Inputs:
#   NNF matrix
#   D-score matrix
#   Boolean: True when current iteration is odd
#   Source image patch (as a matrix of 2D-arrays)
#   Target image (as a matrix of 2D-arrays)
# Output:
#   Updated NNF matrix
#   Updated D matrix
def propagation(NNF, D_matrix, is_odd_iter,
                src_patches, trg_patches):

    # Update the NNF using propagation rule set
    new_nnf = make_updated_NNF(NNF, D_matrix, is_odd_iter)

    # Update the D-scores based on the updated NNF and source/target patches
    updated_D = make_updated_D(new_nnf, src_patches, trg_patches)

    return new_nnf, updated_D

# Input:
#   NNF matrix
#   D-score matrix
#   Boolean: True when current iteration is odd
# Output:
#  An updated version of the NNF matrix (updated using the ruleset
#   defined in section 3.2 of the paper)
def make_updated_NNF(NNF, D_matrix, odd_iter):
    # For finding the indices of the argmin
    simil_comp_arr = stack_shifted(D_matrix, odd_iter)

    # Find the index of the argmin - the following is a mapping of indices to
    # what they mean:
    # 0 -> the current element
    # 1 -> the element in the horizontal direction
    # 2 -> the element in the vertical direction
    # Use nanargmin to ignore NAN values (they used to represent invalid elements)
    argmin_inds = np.nanargmin(simil_comp_arr, axis=2)

    # For updating the NNF on the basis of the argmin
    nnf_stacked = stack_shifted(NNF, odd_iter)

    # Updating the NNF on the basis of the argmin
    new_nnf = np.zeros_like(NNF)
    for i in range(2):
        new_nnf[argmin_inds == i] = nnf_stacked[i]

    # Snip vectors of the NNF so there aren't any vectors that point outside
    # the rectangle, then return this snipped matrix
    return snip_vectors(new_nnf)

# Input:
#   Matrix of vectors that point to (supposed) nearest neighbours
#   Source image patch (as a matrix of 2D-arrays)
#   Target image (as a matrix of 2D-arrays)
#   Size of the patch we are using
# Output:
#   Updated D-matrix (using the given NNF matrix)
def make_updated_D(NNF, src_patches, trg_patches):

    # Finding coordinates of target patches given NNF
    dest_coords = targ_coords(NNF)

    # Array of patches where each patch at [i,j] corresponds to the (supposed) NN patch
    # [i, j] + [x, y], where [x, y] is the NN vector for location [i, j]
    trg_rearranged = trg_patches.choose(dest_coords)

    # Compute the D scores and return them
    return compute_D(src_patches, trg_rearranged)

# Input:
#   Patches of source image in array form
#   Patches of target image in array form - arranged according to some NNF
# Output:
#   2D array of D-scores for each patch
def compute_D(src_patches, trg_rearranged):
    # Going to be using (squared euc-distance / valid area of patch)
    # as measure of patch distance (division ensures that border patches aren't
    # weighted less due to NANs)
    # (avoiding rooting, it's costly and provides no additional benefit)

    # Average the RGB for each pixel in target and source
    # (average and flatten out the color channel (axis 2))
    # Use nanmean to return NAN in the event that a pixel is all NANS (means it is outside)
    src_avg = np.nanmean(src_patches, axis=2)
    trg_avg = np.nanmean(trg_rearranged, axis=2)
    # Last axis will now be the patch contents

    # Take the difference between the patch arrays and square it
    sq_diff = (src_avg - trg_avg) ** 2

    # Add the squared differences together inside each patch (use dot product of 1's)
    # Use nansum to ignore NANs
    patch_sums = np.nansum(sq_diff, axis=2)

    # Figure out the valid area of each patch (number of non-nans)
    unos = sq_diff / sq_diff # 1 where there are valid numbers, NAN where there are nans
    patch_areas = np.nansum(unos, axis=2) # Number of non-nans per patch

    # Divide the total by the total valid patch area (number of non-nans)
    dist = patch_sums / patch_areas

    return dist

# Input:
#   NNF (as a matrix, every element is a 2-array)
# Let:
#   (x_s, y_s) be the coordinates of some patch S in the source image
#   (x_n, y_n) be the x and y components of the NN vector at (x_s, y_s)
#   (x_t, y_t) be the coordinates of some patch T in the target image
#   S and T are (supposed) nearest-neighbour patches
# We note that (z is x or y): z_t = z_s + z_n
# Output:
#   Matrix of destination coordinates [[[x_t, y_t]]]
def targ_coords(nn_matrix):
    coord_mat = coords_of(nn_matrix)
    return coord_mat + nn_matrix

# Input: a 3D array (a matrix of 2-vectors)
# Let:
#   (x, y) be some arbitrary position in the matrix
#   (a, b) be the vector at (x, y)
# Output: new matrix of vectors A = [[[c, d]]] such that:
#       if ((x, y) + (a, b)) is a valid position of the matrix (i.e. in bounds):
#           (c, d) = (a, b)
#       if ((x, y) + (a, b)) not a valid position of the matrix (i.e. out of bounds):
#           (c, d) = T * unit(a, b)
#       Where T is the maximum possible value such that: ((x, y) + T * (a, b)) is a valid position
def snip_vectors(vector_rect):

    # Matrix to return (in float form)
    flt_arr = vector_rect.astype(float)

    # Matrix of positions where target patch coordinates are OUTSIDE the target
    out_matrix = np.negative(in_targ_mask(vector_rect))

    # Matrix of T-values, calculated for every position
    T_matrix = calc_t_val(vector_rect)

    # Matrix of unit vectors
    unit_vecs = unit_vec_rect(vector_rect)

    # Replace vectors that point outside the rectangle in ret_arr
    # with the value specified in the description (c, d) = T * unit(a, b)
    flt_arr[out_matrix] = T_matrix * unit_vecs

    # Return the matrix in integer form (floor the matrix)
    ret_arr = np.floor(flt_arr).astype(int)


    return ret_arr

# Input: a 3D array (a matrix of 2-vectors)
# Let:
#   (x, y) be some arbitrary position in the matrix
#   (a, b) be the vector at (x, y)
# Output: boolean mask B = [[c]] such that:
#       if ((x, y) + (a, b)) is a valid position of the matrix (i.e. in bounds):
#           c = True
#       if ((x, y) + (a, b)) not a valid position of the matrix (i.e. out of bounds):
#           c = False
def in_targ_mask(vector_rect):
    # X, Y components of vector rectangle
    x_rect, y_rect = vector_rect[:, :, 1], vector_rect[:, :, 0]
    # Bounds for X and Y
    height, width = x_rect.shape

    # Mask to return after completion
    ret_mask = np.zeros_like(vector_rect[:, :, 0],
                             dtype=bool)
    # Set positions in mask to True where:
    # Destinations are inside Y bounds (check first component)
    ret_mask[0 <= y_rect < height] = True
    # Destinations are inside X bounds (check second component)
    ret_mask[0 <= x_rect < width] = True

    return ret_mask

# Helper to find the T-value for the else-case described in snip_vectors
# (for all vectors in vector_rect) (returns 0 if any positive T makes the target
# vector go outside)
def calc_t_val(vector_rect):

    # Matrix of unit vectors
    unit_rect = unit_vec_rect(vector_rect)
    # Matrix of coordinates
    coord_mat = coords_of(unit_rect)

    # Splitting the above matrices into x, y components for clarity
    unit_y, unit_x = unit_rect[:,:,0], unit_rect[:,:,1]
    y_coords, x_coords = coord_mat[:,:,0], coord_mat[:,:,1]

    # Getting the coordinates of the rectangle's sides:
    l, d, r, u = side_coords(y_coords)

    # Note: WLOG for x_k = A.x + T * unit(A -> B).x
    # (where A is src coordinate, B is target coordinate):
    # T = (x_k - A.x) / (unit(A -> B).x)

    # T-values for Y
    # To get the out-vectors to intersect with Y = d
    T_d = (d - y_coords) / unit_y
    # To get the out-vectors to intersect with Y = u
    T_u = (u - y_coords) / unit_y

    # T-values for X
    # To get the out-vectors to intersect with X = l
    T_l = (l - x_coords) / unit_x
    # To get the out-vectors to intersect with X = r
    T_r = (r - x_coords) / unit_x

    # Pick T_final = max(T_x, T_y) where:
    #   T_x = (T_r if (unit.x > 0) else T_l)
    #   T_y = (T_u if (unit.y > 0) else T_d)
    # Initialize new array to all zeroes to remedy following edge case:
    #   Denominator of T_udlr is 0:
    #       Means that the NN vector is 0 in at least one direction
    #       Thus no T exists to make the vector intersect a side of the rectangle parallel to it
    #   We set these T values to 0 to indicate that the NN patch in the target is equal
    #      to the same patch in the source
    T_x, T_y = np.zeros_like(T_l), np.zeroes_like(T_d)
    T_x[unit_x > 0] = T_r
    T_x[unit_x < 0] = T_l
    T_y[unit_y > 0] = T_u
    T_y[unit_y < 0] = T_d
    T_final = np.amax(np.dstack((T_x, T_y)),
                      axis=2)
    return T_final

# Input: some 2D-array
# Output: (l, d, r, u) = (x of left, y of down, x of right, y of up)
def side_coords(rect):
    r, u = np.array(rect.shape) - np.array([1, 1])
    return 0, 0, r, u

# Input: a 3D array (a matrix of 2-vectors)
# Let:
#   (x, y) be some arbitrary position in the matrix
#   (a, b) be the vector at (x, y)
# Output: new matrix of vectors A = [[[c, d]]] such that:
#   (c, d) = (unit vector of (a, b))
def unit_vec_rect(vector_rect):
    vec_rec = vector_rect.astype(float)
    # Array such that every element is the magnitude of the corresponding element in vector_rect
    mag_arr = np.dot((vec_rec * vec_rec),
                     np.array([1, 1])) ** 0.5

    # Avoid div 0 errors
    mag_arr[mag_arr == 0] = 1

    # Return the array of unit-vectors
    return (vec_rec / mag_arr)

# Helper to build 3D arrays in the following form (from array arr):
#   Layer 0 = arr
#   Layer 1 = arr_in_dir_of(arr, horizontal dir)
#   Layer 2 = arr_in_dir_of(arr, vertical dir)
# The [i,j]-th element of the resulting array is [Layer0[i,j], Layer1[i,j], Layer2[i,j]]
# Where the directions are as follows for the following iteration types:
#   Odd:
#       Horizontal = Left
#       Vertical = Down
#   Even:
#       Horizontal = Right
#       Vertical = Up
def stack_shifted(arr, is_odd_iter):

    # Map the iteration type to the appropriate H/V directions
    horiz_map = {True : left,
                 False : right}
    vert_map = {True : up,
                False: down}

    # Retrieving the directions based on iteration type
    horz, vert = horiz_map[is_odd_iter], vert_map[is_odd_iter]

    # Return the array in the described format
    return np.dstack((arr,
                      arr_in_dir_of(arr, horz),
                      arr_in_dir_of(arr, vert)))

# Makes a new 2+ dimensional array such that every element [i, j]
# is equal to the [i,j]-th element of arr, shifted over by the given
# direction. Elements that will be shifted to an
# out-of-bounds position will be replaced with kwarg padding.
# Example:
# a = np.array([[1, 2],[3, 4]])
# arr_in_dir_of(a, up) -> array([[NAN, NAN],[1, 2]])
def arr_in_dir_of(arr, shift_dir, padding=np.nan):
    dir_map = {up:down, down:up, left:right, right:left}
    return shift_arr(arr, dir_map[shift_dir], padding=padding)

# Helper for shifting a matrix's elements in one of four directions:
# up, down, left, right. Elements that will be shifted to an
# out-of-bounds position will be replaced with kwarg padding.
# Example:
# a = np.array([[1, 2],[3, 4]])
# shift_arr(a, up) -> array([[3, 4],[NAN, NAN]])
def shift_arr(arr, shift_dir,
              padding=np.nan):

    if shift_dir == up:
        shifted = np.roll(arr, -1, axis=0)
        shifted[-1, :] = padding

    elif shift_dir == down:
        shifted = np.roll(arr, 1, axis=0)
        shifted[0, :] = padding

    elif shift_dir == left:
        shifted = np.roll(arr, 1, axis=1)
        shifted[:, -1] = padding

    elif shift_dir == right:
        shifted = np.roll(arr, -1, axis=1)
        shifted[:, 0] = padding

    return shifted

# Wrapper function to create a matrix of (y, x) - coordinates for a given array:
def coords_of(arr):
    return make_coordinates_matrix(arr.shape[:2])

# Helper for random search
def random_search(w, alpha, NNF_matrix, src_patches, trg_patches):
    # NNF Matrix to return
    ret_NNF = NNF_matrix.copy()

    # Number of iterations the search needs to do
    num_iters = num_iters_needed(w, alpha)

    # D-matrix of the last iteration (initialize to D-score computed from given inputs)
    prev_D_mat = make_updated_D(NNF_matrix, src_patches, trg_patches)

    # Repeat the search for the number of times needed
    for i in range(num_iters):

        # Debugging statement
        print w * (alpha ** i)

        # R_i as described in the paper, one random vector for each vector in the NNF
        R_i_matrix = uni_rand_like(NNF_matrix, -1, 1)

        # The random vector w * alpha^i * R_i (in matrix form for each element of R_i)
        rand_vec_matrix = w * (alpha ** i) * R_i_matrix

        # u_i as described in the paper (NNF vector + random vector) in matrix form
        # Snip vectors so they are bounded inside the image
        u_i_matrix = snip_vectors(NNF_matrix + rand_vec_matrix)

        # Compute the D-matrix for this iteration
        cur_D_mat = make_updated_D(u_i_matrix, src_patches, trg_patches)

        # Update NNF elements to u_i iff the associated D(u_i) < D(u_(i-1))
        # (i.e. D "got better" since the last iteration)
        ret_NNF[cur_D_mat < prev_D_mat] = u_i_matrix

        # Update the D-matrix for the next iteration
        prev_D_mat = cur_D_mat

    return ret_NNF

# Returns the number of iterations needed for w*alpha^i to decay to < 1:
def num_iters_needed(w, alpha):
    # According to Section 3.2, random search terminates when w*alpha^i < 1
    # Solving for i:
    # w*alpha^i < 1
    # =(div by w)-> alpha ^ i < 1 / w
    # =(log both sides)-> i < -log(w)
    # Thus, we can see that the number of iterations needed for random search to terminate is about -log(w)
    return -1 * log(w, alpha)

# Operates like numpy's ones_like or zeroes_like functions, but
# with the values of the matrix as floats uniformly distributed
# in the interval [start, end)
def uni_rand_like(arr, start, end):
    rng_len =  end - start
    rand_arr = np.random.random(size=arr.shape).astype(float)
    return (rand_arr * rng_len) + start


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient
def reconstruct_source_from_target(target, f):
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################


    #############################################

    return rec_source

# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.
def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix

# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops
def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
