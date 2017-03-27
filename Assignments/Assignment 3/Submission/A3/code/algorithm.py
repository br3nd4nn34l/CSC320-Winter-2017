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
flr, cel, rnd = np.floor, np.ceil, np.around

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
                                  propagation_disabled, random_disabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None):

    # Note: after conducting some code tracing and testing,
    # it looks like propagation_disabled and random_disabled
    # are not being set properly: they seem to be set to
    # the negation of what they should be - code seems to be
    # ignoring the fact that (enabled = not disabled) (???)
    # I will be refactoring
    # (propagation_disabled, random_disabled) ->
    #   (propagation_disabled, random_disabled)
    # So the code better reflects what's happening.

    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # Compute D if it is not already provided for us
    if best_D is None:
        best_D = update_D_matrix(f, source_patches, target_patches)

    # Do propagation if we're asked to
    if not propagation_disabled:
        new_f, best_D = propagation(new_f, best_D,
                                    odd_iteration,
                                    source_patches, target_patches)

    # Do random search if we're asked to
    if not random_disabled:
        new_f, best_D = random_search(w, alpha, new_f,
                                      source_patches, target_patches)

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
    new_nnf = make_updated_NNF(NNF, D_matrix, is_odd_iter,
                               src_patches, trg_patches,
                               coords_of(NNF))

    # Update the D-scores based on the updated NNF and source/target patches
    updated_D = update_D_matrix(new_nnf, src_patches, trg_patches)

    return new_nnf, updated_D

# Input:
#   NNF matrix
#   D-score matrix
#   Boolean: True when current iteration is odd
# Output:
#  An updated version of the NNF matrix (updated using the ruleset
#   defined in section 3.2 of the paper)
def make_updated_NNF(NNF, D_matrix, odd_iter, src_patches, trg_patches, coord_mat):

    # We are traversing the matrix in this function as it should be
    # for an even iteration. To perform equivalent operations in the
    # opposite direction(s), we need to reverse the rows and columns
    # of the inputs before recalling this function
    if odd_iter:
        return flip_matrix(make_updated_NNF(
            flip_matrix(NNF),
            flip_matrix(D_matrix),
            False,
            src_patches,
            trg_patches,
            flip_matrix(coord_mat)))

    # The new NNF matrix
    new_nnf = np.zeros_like(NNF)

    # Matrix of D values: use this to decide what NNF vectors are "better"
    D_values = np.zeros_like(D_matrix)

    # Dimensions of matrix (these are the maximum indices)
    y_ub, x_ub = edge_indices(NNF)[2:]

    # Assuming we're starting out in the top
    # left-hand corner (on an EVEN iteration)

    # Step 1) Fill in the top left hand corner with whatever was there before
    #       There is no way that anything up or left is valid,
    #       This is our "base case"
    new_nnf[0, 0] = NNF[0, 0]
    D_values[0, 0] = single_D(NNF[0, 0], coord_mat[0, 0],
                              src_patches, trg_patches)

    # Step 2) Fill in the following rows/cols. There are only two choices here: the current
    # element or the preceding element on the line.

    # Step 2a) Fill in the leftmost COLUMN, going "down"
    for row in range(1, y_ub + 1):

        # Vector and D value for above cell
        vec_above = snip_vector(new_nnf[row - 1, 0],
                                coord_mat[row, 0],
                                new_nnf)
        D_above = single_D(vec_above, coord_mat[row, 0],
                           src_patches, trg_patches)

        # Vector and D value for this cell
        cur_vec = NNF[row, 0]
        cur_D = D_matrix[row, 0]

        # If this D value is better, keep it and this vector
        if cur_D < D_above:
            new_nnf[row, 0] = cur_vec

        # Otherwise, replace with the vector above it and it's associated D-value
        else:
            new_nnf[row, 0] = vec_above

    # Step 2b) Fill in the topmost ROW, going "right"
    for col in range(1, x_ub + 1):

        # Vector and D value for cell to left
        vec_left = snip_vector(new_nnf[0, col - 1],
                               coord_mat[0, col],
                               new_nnf)
        D_left = single_D(vec_left, coord_mat[0, col],
                          src_patches, trg_patches)

        # Vector and D value for this cell
        cur_vec = NNF[0, col]
        cur_D = D_matrix[0, col]

        # If this D value is better, keep this vector
        if cur_D < D_left:
            new_nnf[0, col] = cur_vec

        # Otherwise, replace with the vector to the left
        else:
            new_nnf[0, col] = vec_left

    # Step 3b) Fill in the matrix starting from [1, 1], going right and then going down
    for col in range(1, x_ub + 1): # Going right
        for row in range(1, y_ub + 1): # Going down

            # The vector directly pointed to by this iteration
            # and it's D-value
            cur_vector = NNF[row, col]
            cur_D = D_matrix[row, col]

            # The vector above the current vector and it's D-value
            above_vector = snip_vector(NNF[row - 1, col],
                                       coord_mat[row, col], new_nnf)
            above_D = single_D(above_vector, coord_mat[row, col],
                               src_patches, trg_patches)

            # The vector to the left of the current vector and it's D-value
            left_vector = snip_vector(NNF[row, col - 1],
                                      coord_mat[row, col], new_nnf)
            left_D = single_D(left_vector, coord_mat[row, col],
                              src_patches, trg_patches)

            # Organizing the three things into arrays
            nnf_candidates = np.array([cur_vector, above_vector, left_vector])
            D_vals = np.array([cur_D, above_D, left_D])

            # Pick the minimum of the 3 D-values and assign the minimizing vector
            min_d_ind = np.argmin(D_vals)
            new_nnf[row, col] = nnf_candidates[min_d_ind]

    return snip_vectors(new_nnf)

# Function to reverse the rows and columns of the input array
# (first two axes). Returns a flipped copy
# Example: [[A, B], [C, D]] -> [[D, C], [B, A]]
def flip_matrix(arr):
    cpy = arr.copy()
    row_rev = cpy[::-1, :]
    col_rev = row_rev[:, ::-1]
    return col_rev

# Input:
#   Matrix of vectors that point to (supposed) nearest neighbours
#   Source image patch (as a matrix of 2D-arrays)
#   Target image (as a matrix of 2D-arrays)
#   Size of the patch we are using
# Output:
#   Updated D-matrix (using the given NNF matrix)
def update_D_matrix(NNF, src_patches, trg_patches):

    # Finding coordinates of target patches given NNF
    dest_coords = targ_coords(NNF)

    # Array of patches where each patch at [i,j] corresponds to the (supposed) NN patch
    # [i, j] + [x, y], where [x, y] is the NN vector for location [i, j]
    trg_rearranged = lookup_values(trg_patches, dest_coords)

    # Compute the D scores and return them
    return bulk_D(src_patches, trg_rearranged)

# Input:
#   Patches of source image in array form
#   Patches of target image in array form - arranged according to some NNF
# Output:
#   2D array of D-scores for each patch
# Used for large patch matrices - allows for full leveraging of numpy's abilities
def bulk_D(src_patches, trg_rearranged):
    # Going to be using Mean Squares as measure of patch "distance"
    # (not RMS, rooting is costly and does not change comparisons)

    # Flatten the color channel of the patched images
    # Recast as smaller int to avoid memory error - this will effectively turn NaNs into 0,
    # which allows us to discount out of bound pixels from distance measurements
    src_flatter = color_flatten(src_patches).astype(np.uint16)
    trg_flatter = color_flatten(trg_rearranged).astype(np.uint16)

    # Take the difference between the flattened patch arrays and square it
    # Convert back to float32 - memory concerns are still a thing
    sq_diff = ((src_flatter - trg_flatter) ** 2).astype(np.float32)

    # Set values to NaN where src or target are NaN
    nan_mask = color_flatten(np.logical_or(src_patches == np.nan,
                                           trg_rearranged == np.nan))
    sq_diff[nan_mask] = np.nan

    # Valid area of each Sq-D patch
    valid_area_matrix = non_nan_count_mat(sq_diff)

    # Add the squared differences together inside each patch
    patch_sums = np.nansum(sq_diff, axis=2)

    # Divide the total by the total valid patch area
    # to get the Mean Square matrix
    dist_mat = patch_sums / valid_area_matrix

    return dist_mat

# Input:
#   vec: NNF vector
#   pos: Origin coordinate of the NNF vector
#   src_patches: Patches from source image
#   trg_patches: Patches from target image
# Output:
#   RMS between src_patches[pos] and trg_patches[vec + pos]
def single_D(vec, pos, src_patches, trg_patches):

    # Going to be using Mean Squares as measure of patch "distance"
    # (not RMS, rooting is costly and does not change comparisons)

    trg_coord = vec + pos
    src_patch = src_patches[pos[0], pos[1]]
    trg_patch = trg_patches[trg_coord[0], trg_coord[1]]

    # Flatten the patches so we can treat them as vectors
    src_flat = src_patch.flatten()
    trg_flat = trg_patch.flatten()

    # Take the difference between the flattened patches, square it
    sq_diff = (src_flat - trg_flat) ** 2

    # Figure out the number of non-nan elements in the square diff
    num_non_nans = np.nansum(sq_diff / sq_diff)

    # Sum up the squared differences (ignoring nan values)
    sum_sqd = np.nansum(sq_diff)

    # Divide the sum of squared distances by the total valid patch area
    # to get the Mean Square matrix
    return sum_sqd / num_non_nans


def snip_vector(vec, pos, rect):

    if vector_points_inside(vec, pos, rect):
        return vec
    else:
        # Unit vector
        unit = unit_vec(vec)

        # T value for this unit vector
        T = calculate_single_T(unit, pos, rect)

        # Vector to return = T * unit
        ret_vec = np.array([T, T]) * unit

        # Coordinates of the target patches that the vector points to
        trg_cd = ret_vec + pos

        # Upper / Lower X and Y bounds
        y_lb, x_lb, y_ub, x_ub = edge_indices(rect)

        # Clip the coordinate
        clipped_cds = np.clip(trg_cd,
                              np.array([y_lb, x_lb]),
                              np.array([y_ub, x_ub]))

        # Convert back to an int vector
        return flt_to_int(clipped_cds - pos, rnd)

def unit_vec(vec):
    mag = 1.0 * np.sum((vec * vec)) ** 0.5
    return vec / mag

def calculate_single_T(unit, pos, rect):
    # Splitting unit-vector into x, y components for clarity
    unit_y, unit_x = unit
    y_coord, x_coord = pos

    # Getting the coordinates of the rectangle's sides:
    d, l, u, r = edge_indices(rect)

    # Note: WLOG for x_k = A.x + T * unit(A -> B).x
    # (where A is src coordinate, B is target coordinate):
    # T = (x_k - A.x) / (unit(A -> B).x)

    # T-values for Y
    # To get the vector to intersect with Y = d
    T_d = (d - y_coord) / unit_y
    # To get the vector to intersect with Y = u
    T_u = (u - y_coord) / unit_y

    # T-values for X
    # To get the vector to intersect with X = l
    T_l = (l - x_coord) / unit_x
    # To get the vector to intersect with X = r
    T_r = (r - x_coord) / unit_x

    # Pick T_final = min(T_x, T_y) where:
    #   T_x = (T_r if (unit.x > 0) else T_l)
    #   T_y = (T_u if (unit.y > 0) else T_d)
    T_x, T_y = np.zeros_like(T_l), np.zeros_like(T_d)
    T_x = np.where(unit_x > 0, T_r, T_x)
    T_x = np.where(unit_x < 0, T_l, T_x)
    T_y = np.where(unit_y > 0, T_u, T_y)
    T_y = np.where(unit_y < 0, T_d, T_y)
    T_final = np.amin(np.array((T_x, T_y)))

    return T_final

# Indicates that vec points to a location inside rect
# (when origin is pos)
def vector_points_inside(vec, pos, rect):
    y_lb, x_lb, y_ub, x_ub = edge_indices(rect)
    tgt = vec + pos

    above_lbs = (tgt >= np.array([y_lb, x_lb]))
    below_ubs = (tgt <= np.array([y_ub, x_ub]))

    return np.logical_and.reduce(np.hstack((below_ubs, above_lbs)))


# Input: patch array (dimensions NxMxCxP^2)
# Output: 3D array of shape (N x M x CP^2) (color channel is "flattened" out)
def color_flatten(patches):

    # Fetching dimensions for brevity
    N, M, C, patch_area = patches.shape

    # Don't mess with the original
    ret_arr = patches.copy().reshape((N, M, C * patch_area))

    return ret_arr

# Input: 3D array: Matrix of vectors - vectors may contain NAN values
# Output: 2D array: Matrix such that element [i,j] is the number of non-NAN values in vector [i, j]
def non_nan_count_mat(patches):
    # 1 where there are valid numbers, NAN where there are nans
    unos = patches / patches

    # Count number of non-nan values using nansum
    patch_areas = np.nansum(unos, axis=2)

    # What we want
    return patch_areas

# Input:
#   M by N Matrix [[V_(i, j)]]
#   Y by X Matrix of 2-vectors [[[a, b]]] (3D-array), a in [0, N-1], b in [0, M-1]
# Output:
#   Y by X Matrix where element [a, b] is equal to [[V_(a, b)]]
def lookup_values(src_matrix, index_matrix):
    return src_matrix[split_yx(index_matrix)]

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
    # Casting as int
    return (coord_mat + nn_matrix).astype(int)

# Helper to reconstruct NN matrix given array of target coordinates
# Input: 3D array (matrix of (y, x) coordinates)
# Let:
#   (y_c, x_c) be the position of some 2-tuple in the matrix
#   (y_t, x_t) be the coordinate at input[(y_c, x_c)]
# Output: 3D array of (y_t - y_c, x_t - x_c)
def targ_to_vec(tgt_coords):

    # Matrix of coordinates
    coord_mat = coords_of(tgt_coords)

    # Casting as an int
    return flt_to_int(tgt_coords - coord_mat, rnd)

# Input: matrix where every element is a (y, x) coordinate
#   (this is a 3D array, shape is assumed to be (N x M x 2))
# Output: tuple of (matrix of y-coords, matrix of x-coords)
def split_yx(vector_rect):
    return (vector_rect[:, :, 0], vector_rect[:, :, 1])

# Input:
#   Matrix of vectors (3D array)
# Output:
#   Matrix of vectors in the following form:
#       If a vector did not point outside the rectangle, it remains the same
#       If a vector DID point outside the rectangle, it's length is reduced
#           such that it now targets the nearest edge of the rectangle
#          (directionality is preserved)
def snip_vectors(vector_rect):

    # Matrix to return (in float form)
    flt_arr = vector_rect.astype(float)

    # Matrix of positions where target patch coordinates are OUTSIDE the target
    out_matrix = np.negative(in_targ_mask(vector_rect))

    # Matrix of unit vectors
    unit_vecs = unit_vec_rect(vector_rect)

    # Matrix of T-values, calculated for every position
    T_matrix = calc_t_val(unit_vecs)

    # Replace vectors that point outside the rectangle in ret_arr
    # with the value specified in the description (c, d) = T * unit(a, b)
    flt_arr = np.where(stack_clones(out_matrix, 2),
                       stack_clones(T_matrix, 2) * unit_vecs,
                       flt_arr)

    # NNF Matrix in integer form (floor the matrix)
    # (some vectors may not be completely inside the matrix)
    int_arr = flt_to_int(flt_arr, flr)

    # Coordinates of the target patches that the NNF matrix points to
    trg_cds = targ_coords(int_arr)
    # Upper / Lower X and Y bounds
    y_lb, x_lb, y_ub, x_ub = edge_indices(vector_rect)
    # Clip the coordinates
    clipped_cds = clip_according_to(trg_cds,
                                    np.ones_like(int_arr) * np.array([y_lb, x_lb]),
                                    np.ones_like(int_arr) * np.array([y_ub, x_ub]))
    # Convert back to vectors
    ret_arr = targ_to_vec(clipped_cds)

    return ret_arr

# Input:
#   arr: N-D array of integers (I)
#   low_bd: N-D array of integers (L)
#   up_bd: N-D array of integers (U)
# Output:
#   Array A such that I[i, ...] in the range [L[i, ...], U[i, ...]]
def clip_according_to(arr, low_bd, up_bd):
    # How to pack and unpack arrays
    pack_method = "F"

    # Flatten out the input arrays for easier reasoning
    flat_E, flat_L, flat_U = arr.flatten(order=pack_method), \
                             low_bd.flatten(order=pack_method), \
                             up_bd.flatten(order=pack_method)

    # Use clip function
    clipped = np.clip(flat_E, flat_L, flat_U)

    # Reshape the choices in the same shape as the original
    reshaped = clipped.reshape(arr.shape, order=pack_method)

    return reshaped

# Helper to find the T-value needed to extend the unit-vectors in
# unit_vecs to touch the sides of the rectangle
def calc_t_val(unit_vecs):

    # Splitting unit-vector and coordinate matrices into
    # x, y components for clarity
    unit_y, unit_x =  split_yx(unit_vecs)
    y_coords, x_coords = split_yx(coords_of(unit_vecs))

    # Getting the coordinates of the rectangle's sides:
    d, l, u, r = edge_indices(unit_vecs)

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

    # Pick T_final = min(T_x, T_y) where:
    #   T_x = (T_r if (unit.x > 0) else T_l)
    #   T_y = (T_u if (unit.y > 0) else T_d)
    # Initialize new array to all zeroes to remedy following edge case:
    #   Denominator of T_udlr is 0:
    #       Means that the NN vector is 0 in at least one direction
    #       Thus no T exists to make the vector intersect a side of the rectangle parallel to it
    #   We set these T values to 0 to indicate that the NN patch in the target is equal
    #      to the same patch in the source
    T_x, T_y = np.zeros_like(T_l), np.zeros_like(T_d)
    T_x = np.where(unit_x > 0, T_r, T_x)
    T_x = np.where(unit_x < 0, T_l, T_x)
    T_y = np.where(unit_y > 0, T_u, T_y)
    T_y = np.where(unit_y < 0, T_d, T_y)
    T_final = np.amin(np.dstack((T_x, T_y)),
                      axis=2)

    return T_final

# Input: a 3D array (a matrix of 2-vectors)
# Let:
#   (x, y) be some arbitrary position in the matrix
#   (a, b) be the vector at (x, y)
# Output: new matrix of vectors A = [[[c, d]]] such that:
#   (c, d) = (unit vector of (a, b))
def unit_vec_rect(vector_rect):

    # Convert to floats for easy reasoning
    vec_rec = vector_rect.astype(float)
    # Array such that every element is the magnitude of the corresponding element in vector_rect
    mag_arr = np.dot((vec_rec * vec_rec),
                     np.array([1, 1])) ** 0.5

    # Avoid div 0 errors
    mag_arr[mag_arr == 0] = 1

    # Return the array of unit-vectors
    return (vec_rec / stack_clones(mag_arr, 2))

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
    # Where the vectors point to
    trg_cds = targ_coords(vector_rect)
    # X, Y components of destination components
    y_rect, x_rect = split_yx(trg_cds)
    # Bounds for X and Y
    height, width = x_rect.shape

    # Set positions in mask to True where:
    # Destinations are inside Y and X bounds (check first component)
    ret_mask = np.logical_and.reduce((0 <= y_rect,
                                     (y_rect < height),
                                     (0 <= x_rect),
                                     (x_rect < width)))

    return ret_mask

# Input: some (2+)D-array
# Output: (d, l, u, r) = (y of down side,
#                         x of left side,
#                         y of up side,
#                         x of right side)
def edge_indices(rect):
    u, r = np.array(rect.shape[:2]) - np.array([1, 1])
    return 0, 0, u, r

# Helper to build 4D arrays in the following form (from array arr):
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
# Maxed empties indicates whether we should fill the "empty" slots with the maximum value
# of arr's dtype. Otherwise, we fill it with the minimum value of arr's dtype.
def stack_shifted(arr, is_odd_iter, maxed_empties):

    # Map the iteration type to the appropriate H/V directions
    horiz_map = {True : left,
                 False : right}
    vert_map = {True : up,
                False: down}

    # Retrieving the directions based on iteration type
    horz, vert = horiz_map[is_odd_iter], vert_map[is_odd_iter]

    # Return the array in the described format (float to preserve NaNs)
    return np.stack((arr,
                     arr_in_dir_of(arr, horz, maxed_empties),
                     arr_in_dir_of(arr, vert, maxed_empties)), axis=-1)

# Makes a new 2+ dimensional array such that every element [i, j]
# is equal to the [i,j]-th element of arr, shifted over by the given
# direction. Elements that will be shifted to an
# out-of-bounds position will be replaced with padding
# Example:
# a = np.array([[1, 2],
#               [3, 4]])
# arr_in_dir_of(a, up) -> array([[padding, padding],
#                                [1, 2]])
def arr_in_dir_of(arr, shift_dir, padding):

    dir_map = {up:down, down:up, left:right, right:left}
    return shift_arr(arr, dir_map[shift_dir], padding)

# Helper for shifting a matrix's elements in one of four directions:
# up, down, left, right. Elements that will be shifted to an
# out-of-bounds position will be replaced with kwarg padding.
# Example:
# a = np.array([[1, 2],[3, 4]])
# shift_arr(a, up) -> array([[3, 4],[NAN, NAN]])
def shift_arr(arr, shift_dir,
              padding):

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

# Wrapper function to create a matrix of
# (y, x) - coordinates for a given array (only 2D coordinates):
def coords_of(arr):
    return make_coordinates_matrix(arr.shape[:2])

# Inputs:
#   w, alpha: described in paper
#   NNF matrix
#   Patches of source image
#   Patches of target image
# Outputs:
#   NNF matrix updated according to the random search rule-set defined in section 3.2
#   D-score matrix computed using the above NNF matrix
def random_search(w, alpha, NNF_matrix, src_patches, trg_patches):
    # NNF Matrix to return
    ret_NNF = NNF_matrix.copy()

    # Number of iterations the search needs to do
    num_iters = num_iters_needed(w, alpha)

    # D-matrix of best D's (initialize to D-score computed from given inputs)
    best_D_mat = update_D_matrix(NNF_matrix, src_patches, trg_patches).copy()

    # Repeat the search for the number of times needed
    for i in range(num_iters):

        # R_i as described in the paper, one random vector for each vector in the NNF
        R_i_matrix = uni_rand_like(NNF_matrix, -1, 1)

        # The random vector w * alpha^i * R_i (in matrix form for each element of R_i)
        rand_vec_matrix = w * (alpha ** i) * R_i_matrix

        # u_i as described in the paper (NNF vector + random vector) in matrix form
        # Handle vectors so they are bounded inside the image
        u_i_matrix = snip_vectors(NNF_matrix + rand_vec_matrix)

        # Compute the D-matrix for this iteration
        cur_D_mat = update_D_matrix(u_i_matrix, src_patches, trg_patches)

        # Update NNF elements to u_i iff the associated D is better than the best
        ret_NNF = np.where(stack_clones(cur_D_mat < best_D_mat, 2),
                           u_i_matrix, # Replace with new vector newer D is smaller
                           ret_NNF) # Keep old vector if newer D is bigger

        # Update best D matrix with the better D values
        best_D_mat = np.where(cur_D_mat < best_D_mat,
                              cur_D_mat,  # Replace with D if smaller
                              best_D_mat)  # Keep old D if newer D is bigger

    # Compute the D-score for the NNF we are about to return
    final_D = update_D_matrix(ret_NNF, src_patches, trg_patches)

    return ret_NNF, final_D

# Returns the integer number of iterations needed
# for w*alpha^i to decay to < 1:
def num_iters_needed(w, alpha):
    # According to Section 3.2, random search terminates when w*alpha^i < 1
    # Solving for i:
    # w*alpha^i < 1
    # =(div by w)-> alpha ^ i < 1 / w
    # =(log both sides)-> i < -log(w)
    # Thus, we can see that the number of iterations needed for random search to terminate is about -log(w)

    # Return an int, this will be used for loops
    return int(-1 * log(w, alpha)) + 2

# Input:
#   exclusions: N-D array of integers (E)
#   low_bd: N-D array of integers (L)
#   up_bd: N-D array of integers (U)
# Output:
#   Array A such that A[i, ...] not in [E[i, ...], E[i, ...] + 1]
#   but is otherwise uniformly distributed in the range
#   [L[i, ...], U[i, ...])
def exclusive_rand(exclusions, low_bd, up_bd):

    # How to pack and unpack arrays
    pack_method = "F"

    # Flatten out the input arrays for easier reasoning
    flat_E, flat_L, flat_U = exclusions.flatten(order=pack_method), \
                             low_bd.flatten(order=pack_method), \
                             up_bd.flatten(order=pack_method)

    # Uniformly dist in the range [L, E)
    lower = flt_to_int(uni_rand_like(flat_E, flat_L, flat_E),
                       flr)
    # Uniformly dist in the range (E, U)
    upper = flt_to_int(uni_rand_like(flat_E, flat_E + 1, flat_U),
                       rnd)

    # %s of ranges made up from lower half
    lower_dist = (flat_E - flat_L).astype(float)
    whole_dist = (flat_U - flat_L).astype(float)
    lower_pcnt = lower_dist / whole_dist

    # Make decisions on which range to pick from based on this matrix
    choices = uni_rand_like(lower_pcnt, 0, 1)

    # Make our choices
    # Pick lower number when choices < lower_pcnt
    # Pick upper number otherwise
    fair_choice = np.where(choices < lower_pcnt,
                           lower,
                           upper)

    # Reshape the choices in the same shape as the original
    reshaped = fair_choice.reshape(exclusions.shape, order=pack_method)

    return reshaped

# Operates like numpy's ones_like or zeroes_like functions, but
# with the values of the matrix as floats uniformly distributed
# in the interval [start, end)
def uni_rand_like(arr, start, end):
    rng_len =  end - start
    rand_arr = np.random.random(size=arr.shape).astype(np.float32)
    return (rand_arr * rng_len) + start

# Input:
#   arr: Array of dimension N
#   depth: Number of times to stack arr on top of itself, along a new axis
# Output:
#   Array of dimension N+1 (input array stacked depth times through last axis)
def stack_clones(arr, depth):
    tup_to_stack = tuple([arr] * depth) # convert repeated list to tuple
    return np.stack(tup_to_stack, axis=-1) # stack along last axis

# Input:
#   N-D array of floats
#   Rounding method
# Output:
#   N-D array of integers (determined by rounding method)
def flt_to_int(arr, method):
    return method(arr).astype(int)

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

    # Matrix such that element [x, y] = (x, y) + f(x, y)
    tgt_coords = targ_coords(f)

    # Look up pixels in target
    rec_source = lookup_values(target, tgt_coords)

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
