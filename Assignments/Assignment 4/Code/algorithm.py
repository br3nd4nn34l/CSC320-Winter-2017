# CSC320 Winter 2017
# Assignment 4
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
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')

# MY STUFF
# Importing heapify from heapq
from heapq import heapify

# For accessing the different elements of the tuples
priority, counter, displacement = 0, 1, 2


# Input:
#   coll: some collection of same-type objects
#   item: some item with the same type as those in coll
# Adds item to coll
def safe_add(coll, item):
    if type(coll) is list:
        coll += [item]
    if type(coll) is dict:
        coll[item[0]] = item
    if type(coll) is set:
        coll.add(item)

# Input:
#   arr: 2D python list
#   inds: tuple of POSITIVE indices we want to access on the python list
# Output:
#   Tries to access arr[inds]. If this fails, returns None.
def safe_lst_lookup(lst, inds):
    if inds[0] > -1 and inds[1] > -1:
        try:
            return lst[inds[0]][inds[1]]
        except IndexError:
            return None
    else:
        return None

# Input:
#   vect_arr: an array of NN vectors
#   pos_arr: an array of the NN vector's respective positions
#   rect: some (2+)D matrix
# Output:
#   Boolean array A such that:
#       A[i] = whether target coordinate i (vect_arr[i] + pos_arr[i]) is
#       inside the bounds of rect
def in_vectors(vect_arr, pos_arr, rect):
    y_lb, x_lb, y_ub, x_ub = edge_indices(rect)

    tgt = vect_arr + pos_arr

    above_lbs = (tgt >= np.array([y_lb, x_lb])).min(axis=1)

    below_ubs = (tgt <= np.array([y_ub, x_ub])).min(axis=1)

    return np.vstack((above_lbs, below_ubs)).min(axis=0)

# Input: some (2+)D-array
# Output: np.array(d, l, u, r) = (
# y of down side,
# x of left side,
# y of up side,
#  x of right side
# )
def edge_indices(rect):

    # Only take the first two shape values (these will the the
    # height/width of the rectangle)
    u, r = np.array(rect.shape[:2]) - np.array([1, 1])

    return np.array([0, 0, u, r])

# Input:
#   M by N Matrix [[V_(i, j)]]
#   Y by X Matrix of 2-vectors [[[a, b]]] (3D-array), a in [0, N-1], b in [0, M-1]
# Output:
#   Y by X Matrix where element [a, b] is equal to [[V_(a, b)]]
def lookup_values(src_matrix, index_matrix):
    return src_matrix[split_yx(index_matrix)]

# Input: matrix where every element is a (y, x) coordinate
#   (this is a 3D array, shape is assumed to be (N x M x 2))
# Output: tuple of (matrix of y-coords, matrix of x-coords)
def split_yx(vector_rect):
    return (vector_rect[..., 0], vector_rect[..., 1])

# Input:
#   List of tuples [(A, B, ...), (C, D, ...), ...]
# Output:
#   List of "unzipped" tuples [(A, C, ...), (B, D, ...), ...]
def reverse_zip(lst_of_tups):
    # Cool hack I discovered during the summer at work,
    # can't remember where I found it (probably StackEx),
    # but it's burned into my brain
    return zip(*lst_of_tups)

# Input:
#   cur_pos: 2-tuple indicating the position of the heap we want to update in the image
#   heap_matrix: the matrix of NNF heaps
#   dup_matrix: the matrix of items we're using to check duplicate displacements against
#   src_heap: some NNF heap we're trying to improve heap_matrix[cur_pos] with
#   src_patches: patches of the source image
#   trg_patches: patches of the target image
# Improves heap_matrix[cur_pos] and updates dup_matrix[cur_pos] using the NN vectors
# contained in src_heap
def improve_heap(cur_pos, heap_matrix, dup_matrix, src_heap,
                 src_patches, trg_patches):

    # Only do this if the src_heap is valid
    if src_heap is not None:

        # Accessing the stuff we want to update
        cur_heap = heap_matrix[cur_pos[0]][cur_pos[1]]
        cur_dups = dup_matrix[cur_pos[0]][cur_pos[1]]

        # List of tuples with their D-values updated for the
        # new position they will occupy
        updated_tups = update_tups(cur_pos, src_heap,
                                   src_patches, trg_patches)

        # For each tuple, try to improve the current heap with it
        for tup in updated_tups:

            # Need to convert displacement from numpy array
            # to a tuple for duplicate logging
            disp = tuple(tup[displacement])

            # If the displacement hasn't already been logged
            if disp not in cur_dups:

                # Try it out on the heap
                heappushpop(cur_heap, tup)

                # Add it to the list of duplicates so we don't
                # reconsider it
                safe_add(cur_dups, disp)

# Input:
#   cur_pos: the new starting position for the vectors from src_heap
#   src_heap: some NNF heap
#   src_patches: patches of the source image
#   trg_patches: patches of the target image
# Output:
#   List of form [(priority, displacement, ctr)],
#   where priority (D-score) has been updated accordingly
#   as the vectors changed position
def update_tups(cur_pos, heap,
                src_patches, trg_patches):

    # Get the vectors in the heap (they are stored as np arrays)
    nn_vec_arr = np.array(map(lambda tup: tup[displacement],
                              heap))

    # Get the D-values of the above vectors if they were to start
    # from the current position
    d_val_arr = calculate_Ds(nn_vec_arr, cur_pos,
                             src_patches, trg_patches)

    # Get the counter values of these vectors
    ctr_lst = map(lambda tup: tup[counter],
                  heap)

    # Zip the three arrays together back into a list of
    # tuples of the form [(priority, ctr, displacement)]
    return zip(-d_val_arr, ctr_lst, nn_vec_arr)

# Input:
#   vector_arr: array of 2D NN vectors
#   pos: the starting position of the vectors
#   src_patches: patches from source image
#   trg_patches: patches from target image
# Output:
#   D-value for each vector in vector_arr, ordered in the same fashion
def calculate_Ds(vector_arr, pos, src_patches, trg_patches):

    # Which vectors point outside the target
    outside_target = np.negative(in_vectors(
        vector_arr, pos, src_patches))

    # Setting the vectors that point outside the target to
    # 0,0 so they don't give OOB errors
    good_vecs = vector_arr.copy()
    good_vecs[outside_target] = np.array([0, 0])

    # Where the above vectors end up in the target image
    target_positions = good_vecs + pos

    # Selecting the patches from the source and target
    src_selection = src_patches[pos[0], pos[1]]
    trg_selection = trg_patches[target_positions[:, 0],
                                target_positions[:, 1]]

    # Set the out-vector's patches to all NaN
    trg_selection[outside_target] = np.nan

    # Taking the square difference between the source and the target
    sq_diff_of_sels = (src_selection - trg_selection) ** 2

    # Setting NAN differences to highest possible value
    sq_diff_of_sels[np.isnan(sq_diff_of_sels)] = 255 ** 2

    # Sum up the values across color channels and inside patches
    sum_sd_of_sels = np.sum(np.sum(sq_diff_of_sels, axis=1), axis=1)

    return sum_sd_of_sels


# Uses numpy to quickly calculate the D-values of an
# entire NNF matrix
def multiple_D(nnf, src_patches, trg_patches):
    # Matrix of the target points of the nnf
    trg_coords = nnf + coords_of(nnf)

    # Target patches rearranged in terms of target points
    trg_rearr = lookup_values(trg_patches, trg_coords)

    # Using measurement described in single_D (sum of square diffs
    # averaged over number of valid pixels)
    sq_diffs = (src_patches - trg_rearr) ** 2

    # Set to maximum possible square difference if the cell is NaN
    # to increase the D-score for patches with invalid elements
    sq_diffs[np.isnan(sq_diffs)] = 255 ** 2

    # Summing across color channels and then inside patches,
    # use nansum to ignore invalid values
    sq_diff_sums = np.nansum(np.nansum(sq_diffs,
                                       axis=2), axis=2)

    # Return sum
    return sq_diff_sums

# Does propagation and random search for a pixel located at cur_pos
def per_pixel_improvement(cur_pos, prop_enabled, random_enabled,
                          src_patches, trg_patches,
                          f_heap, f_coord_dictionary,
                          alpha, w,
                          odd_iteration):

    if prop_enabled:

        # Offset if the iteration is even is Up, Left = [-1, -1]
        # (since coordinates are reversed in images)
        offset = -1
        if odd_iteration:
            # The offset is the opposite for even iterations
            offset = 1

        # Get the positions offset vertically and horizontally if possible
        # Example: on an even iteration
        vert_pos = (cur_pos[0] + offset, cur_pos[1])
        horz_pos = (cur_pos[0], cur_pos[1] + offset)

        # Get the NNF heaps located at the aforementioned positions
        vert_heap = safe_lst_lookup(f_heap, vert_pos)
        horz_heap = safe_lst_lookup(f_heap, horz_pos)

        # Try to improve the heap with the vertical heap
        improve_heap(cur_pos,
                     f_heap, f_coord_dictionary,
                     vert_heap,
                     src_patches, trg_patches)

        # Try to improve the heap with the horizontal heap
        improve_heap(cur_pos,
                     f_heap, f_coord_dictionary,
                     horz_heap,
                     src_patches, trg_patches)

    if random_enabled:

        # TODO: figure out how to do random search with OOB vectors
        # BOUND SOLUTION IN A3 made this fast!
        # TRY RADHIKA'S SOLUTION - THAT WAS PRETTY FAST AND SHOULD WORK HERE

        random_search(cur_pos, f_heap, f_coord_dictionary, alpha, w)

# Does random search to attempt to improve the NN
# vector heap located at cur_pos
def random_search(cur_pos, f_heap, f_coord_dictionary, alpha, w):
    pass

# ASSIGNMENT FUNCTIONS BELOW THIS POINT

# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
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
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
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
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function
def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################

    # These are the dimensions of the image
    all_dims = edge_indices(source_patches) + 1

    # Rows and columns of the image
    num_rows, num_cols = all_dims[2:]

    # Go through each pixel
    for row in range(num_rows):
        for col in range(num_cols):

            # Try to improve each pixel with the methodology
            # described in the paper
            per_pixel_improvement(
                (row, col), propagation_enabled, random_enabled,
                source_patches, target_patches,
                f_heap, f_coord_dictionary,
                alpha, w, odd_iteration
            )


    #################################################

    return global_vars

# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders
def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # k in the paper
    k = f_k.shape[0]

    # Y, X-components of the nnf vectors
    y_comps, x_comps = split_yx(f_k)

    # Compute D-values for all vectors in the function
    # For-loops acceptable because k is assumed to be small,
    # majority of computation is done in numpy anyways
    # Multipy D_matr by -1 to make the heap behave like a max-heap
    D_matr = np.empty((f_k.shape[:3]))
    for i in range(k):
        D_matr[i] = -multiple_D(f_k[i],
                                source_patches,
                                target_patches)

    # Getting end indexes of the image
    rows, cols = edge_indices(source_patches)[2:] + 1

    # 4D array where:
    #   Axis 0: k - value
    #   Axis 1: y axis
    #   Axis 2: x axis
    #   At arr[k, x, y], we have the 4-array [D, counter, f_y, f_x] (f_i is the i-component of f)
    combined = np.empty((f_k.shape[0], f_k.shape[1], f_k.shape[2], 4))
    # Set first component to priority (D-value)
    combined[:, :, :, 0] = D_matr[:, :, :]
    # Set second component to unique counter (tie breaker)
    combined[:, :, :, 1] = np.arange((rows * cols * k)).reshape((k, rows, cols))
    # Set last components to nnf
    combined[:, :, :, 2] = y_comps[:, :, :]
    combined[:, :, :, 3] = x_comps[:, :, :]

    # Rearrange such that each pixel is an array of length k
    # with each element being an array of length 4
    combined = combined.swapaxes(0, 1).swapaxes(2, 1)

    # Use some clever mapping to quickly transform the above numpy array
    # into a 3D list of tuples
    f_heap = map(lambda row:
                 map(lambda cells:
                     map(lambda cell:
                         (cell[priority],
                          int(cell[counter]),
                          np.array(cell[displacement:]).astype(int)),
                         cells),
                     row),
                 combined)

    # Heapify each cell in the above heap (can't map because
    # heapify is in-place)
    for row in range(rows):
        for col in range(cols):
            heapify(f_heap[row][col])

    # Make the 3D list for duplicate checking
    f_coord_dictionary = map(lambda row:
                             map(lambda tup_lst:
                                 set(map(lambda tup:
                                         tuple(tup[displacement]),
                                         tup_lst)),
                                 row),
                             f_heap)

    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#
def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # Using map to convert each tuple into list
    # Ignore counter because it's useless
    nested_lsts = map(lambda row:
                      map(lambda lst_of_tup:
                          map(lambda tup:
                              [-tup[priority],
                               tup[displacement][0],
                               tup[displacement][1]],
                              lst_of_tup),
                          row),
                      f_heap)

    # Convert the above multi-D list to a numpy array
    numpy_arr = np.array(nested_lsts)

    # Re-arrange array such that K is the first axis
    numpy_arr = numpy_arr.swapaxes(2, 0).swapaxes(1, 2)

    # D_k is the first layer of the above array
    D_k = numpy_arr[..., priority]

    # f_k is the remaining layer of the above array
    f_k = np.empty((D_k.shape[0], D_k.shape[1], D_k.shape[2], 2))
    f_k[..., 0] = numpy_arr[..., 1]
    f_k[..., 1] = numpy_arr[..., 2]

    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################


    #############################################

    return denoised


#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################


#############################################



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

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################


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


# Wrapper function to create a matrix of
# (y, x) - coordinates for a given array (only 2D coordinates):
def coords_of(arr):
    return make_coordinates_matrix(arr.shape[:2])

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
