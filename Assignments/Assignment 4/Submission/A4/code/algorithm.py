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
from heapq import heappush, heappushpop, nlargest, heapify
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


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


    #############################################

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
    D_matr = np.empty((f_k.shape[:3]))
    for i in range(k):
        D_matr[i] = multiple_D(f_k[i],
                               source_patches,
                               target_patches)

    # Multiply D_matr by -1 so the heap behaves like a max-heap
    # (biggest -> most negative will be smallest now)
    D_matr *= -1

    # 4D array where:
    #   Axis 0: k - value
    #   Axis 1: y axis
    #   Axis 2: x axis
    #   At arr[k, x, y], we have the 4-array [D, counter, f_y, f_x] (f_i is the i-component of f)
    combined = np.empty((f_k.shape[:3], 4))
    # Set first component to priority (D-value)
    combined[:, :, :, 0] = D_matr[:, :, :]
    # Set second component to unique counter (tie breaker)
    combined[:, :, :, 1] *= (np.arange(k))
    # Set last components to nnf
    combined[:, :, :, 2] = y_comps[:, :, :]
    combined[:, :, :, 3] = x_comps[:, :, :]

    rows, cols = edge_indices(f_k)[2:]

    # f_heap and f_coord_dictionary is the same size as the rectangle
    f_heap = [[None] * cols] * rows
    f_coord_dictionary = [[None] * cols] * rows


    for row in range(rows):
        for col in range(cols):
            # Make each element in f_heap a heap
            # List of 4-tuples of the current location
            lst_of_tups = map(lambda sub_arr: tuple(sub_arr),
                              list(combined[:, row, col, :]))
            f_heap[row, col] = heapify(lst_of_tups)

            # Make each element in f_coord_dictionary a set of seen
            # displacements
            # List of 2-tuples of displacements
            lst_of_tups = map(lambda sub_arr: tuple(sub_arr),
                              list(combined[:, row, col, 1:3]))

            f_coord_dictionary[row, col] = set(lst_of_tups)

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
def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    # Getting dimensions of return matrix
    rows, cols, k = len(f_heap), len(f_heap[0]), len(f_heap[0][0])

    # Initializing f_k and D_k
    f_k = np.empty((k, rows, cols, 2))
    D_k = np.empty((k, rows, cols))

    # Look at each heap in f_k
    for row in range(rows):
        for col in range(cols):
            # Access the heap
            lst_of_tups = f_heap[row][col]

            # Sort the heap according to D-value (most negative ->
            # biggest original D will be first, so reverse to get the
            # smallest originals first) (builtin sort cheaper than explicit
            # heap-sort)
            sorted_by_d = reversed(sorted(lst_of_tups, key=lambda tup: tup[0]))

            # Extract the D-values and vectors out of the sorted list
            # Ignore counter - it's useless outside the heap
            ds, f_ys, f_xs  = reverse_zip(sorted_by_d)[:-1]

            # Store the values in the arrays they belong inside
            D_k[:, row, col] = (-1 * np.array(ds)) # Get the original D's back
            f_k[:, row, col, 0] = np.array(f_ys)
            f_k[:, row, col, 1] = np.array(f_xs)

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

""" WRAPPER / SHORTHAND FUNCTIONS """
# Input:
#   List of tuples [(A, B, ...), (C, D, ...), ...]
# Output:
#   List of "unzipped" tuples [(A, C, ...), (B, D, ...), ...]
def reverse_zip(lst_of_tups):
    # Cool hack I discovered during the summer at work,
    # can't remember where I found it (probably StackEx),
    # but it's burned into my brain
    return zip(*lst_of_tups)

# Function to reverse the rows and columns of the input array
# (first two axes). Returns a flipped copy
# Example: [[A, B], [C, D]] -> [[D, C], [B, A]]
def flip_matrix(arr):
    if arr is not None:
        cpy = arr.copy()
        row_rev = cpy[::-1, :]
        col_rev = row_rev[:, ::-1]
        return col_rev
    else:
        return None

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

# Input: some (2+)D-array
# Output: (d, l, u, r) = (y of down side,
#                         x of left side,
#                         y of up side,
#                         x of right side)
def edge_indices(rect):

    # Only take the first two shape values (these will the the
    # height/width of the rectangle)
    u, r = np.array(rect.shape[:2]) - np.array([1, 1])
    return np.array([0, 0, u, r])

# Wrapper function to create a matrix of
# (y, x) - coordinates for a given array (only 2D coordinates):
def coords_of(arr):
    return make_coordinates_matrix(arr.shape[:2])

# Input:
#   arr: Array of dimension N
#   depth: Number of times to stack arr on top of itself, along a new axis
# Output:
#   Array of dimension N+1 (input array stacked depth times through last axis)
def stack_clones(arr, depth):
    tup_to_stack = tuple([arr] * depth) # convert repeated list to tuple
    return np.stack(tup_to_stack, axis=-1) # stack along last axis

# Returns the integer number of iterations needed
# for w*alpha^i to decay to < 1:
def num_iters_needed(w, alpha):
    # According to Section 3.2, random search terminates when w*alpha^i < 1
    # Solving for i:
    # w*alpha^i < 1
    # =(div by w)-> alpha ^ i < 1 / w
    # =(log both sides)-> i < -log_alpha (w)
    # =(mult by -1) -> -i > log_alpha(w)

    # Thus, we can see that the number of iterations needed for random search to terminate is about -log(w)

    # Return an int, this will be used for loops
    return int(-1 * log(w, alpha)) + 1

# Input:
#   N-D array of floats
#   Rounding method
# Output:
#   N-D array of integers (determined by rounding method)
def flt_to_int(arr, method):
    return method(arr).astype(int)

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
    trg_cds = vector_rect + coords_of(vector_rect)

    # X, Y components of destination components
    y_rect, x_rect = split_yx(trg_cds)

    # Bounds for X and Y
    height, width = edge_indices(vector_rect)[2:]

    # Set positions in mask to True where:
    # Destinations are inside Y and X bounds (check first component)
    ret_mask = np.logical_and.reduce((0 <= y_rect,
                                     (y_rect < height),
                                     (0 <= x_rect),
                                     (x_rect < width)))
    return ret_mask
"""END OF WRAPPER / SHORTHAND FUNCTIONS"""

"""FUNCTIONS FOR COMPUTING D-VALUES AND BEST-VECTORS"""
# Uses numpy to quickly calculate the D-values of an
# entire NNF matrix. If a vector f in nnf is out of bounds,
# sets D(f) to highest possible value (255^2 * patch size)
def multiple_D(nnf, src_patches, trg_patches):

    # Matrix of the target points of the nnf
    trg_coords = nnf + coords_of(nnf)

    # Matrix that indicates invalid vectors (those that point OUTSIDE
    # the rectangle)
    out_vec_mask = stack_clones(np.negative(in_targ_mask(nnf)), 2)

    # Set target coordinates to something definitely in the image
    # so they don't cause lookup errors in the next step
    trg_coords[out_vec_mask] = 0

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

    # Set the out-vectors' D-values to highest possible D-score
    # (they don't point to valid locations!)
    window_size = np.product(np.array(src_patches.shape[2:]))
    sq_diff_sums[out_vec_mask] = window_size * (255 ** 2)

    # Return sum
    return sq_diff_sums

# Input:
#   vector_arr: array of 2D NN vectors
#   pos: the starting position of the vectors
#   src_patches: patches from source image
#   trg_patches: patches from target image
# Output:
#   D-value for each vector in vector_arr, ordered in the same fashion
def calculate_Ds(vector_arr, pos, src_patches, trg_patches):
    target_positions = vector_arr + pos

    src_selection = src_patches[pos[0], pos[1]]
    trg_selection = trg_patches[target_positions[:, 0],
                                target_positions[:, 1]]

    sq_diff_of_sels = (src_selection - trg_selection) ** 2

    sq_diff_of_sels[np.isnan(sq_diff_of_sels)] = 255 ** 2

    sum_sd_of_sels = np.sum(np.sum(sq_diff_of_sels, axis=1), axis=1)

    return sum_sd_of_sels

# Picks the vector with the lowest D-value in vector_arr
def best_vector(vector_arr, pos, src_patches, trg_patches):

    # D-value of each vector (in order of the vector array)
    D_vals = calculate_Ds(vector_arr, pos, src_patches, trg_patches)

    # Index of the best D value
    min_D_ind = np.argmin(D_vals)

    # Return the vector with the best D-value and the d-value
    return vector_arr[min_D_ind], D_vals[min_D_ind]
"""END OF FUNCTIONS FOR COMPUTING D-VALUES AND BEST-D VECTORS"""


"""FUNCTIONS TO DEAL WITH VECTOR SNIPPING"""
"""VECTOR SNIPPING - "SNIPPING" A VECTOR ONCE IT HITS THE EDGE OF A BOUNDING RECTANGLE"""
# Input:
#   vec: some 2D vector
#   pos: some 2D vector
#   rect: some (2D+) array
# Output:
#   If vec exceeds the bounds of rect, cuts vec off at edge
#   that it intersects. This is so we can preserve the
#   directionality of vec without violating target coord
#   containment constraints
def snip_vector(vec, pos, rect):

    if vector_points_inside(vec, pos, rect):
        return vec
    else:
        # Unit vector
        mag = 1.0 * np.sum((vec * vec)) ** 0.5
        unit = vec / mag

        # T value for this unit vector
        T = calculate_T(unit, pos, rect)

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


def calculate_T(unit, pos, rect):
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
"""END OF FUNCTIONS TO DEAL WITH VECTOR SNIPPING"""


"""FUNCTIONS FOR RANDOM SEARCH"""
# Function to conduct random search for some vector.
# Inputs:
#   pos: Origin position of the vector in question
#   vec: some NN vector
#   D_of_vec: the D-value of vec
#   bound_matr: multi-dimensional array of pre-generated
#       R_i values, one for each i, stacked on top of each other.
#   src_patches: patches of the source image
#   trg_patches: patches of the target image
# Outputs:
#   A vector with a better D-score and the corresponding D-value
def rand_search(pos, vec, bound_matr, src_patches, trg_patches):

    # This is the target coordinate of vec
    # (where v_0 in the paper ends up)
    tgt_coord = vec + pos

    # Component bounds for vectors originating from tgt_coord
    # First Axis: the i in R_i, Second Axis: bounds for that i
    bounds = bound_matr[tgt_coord[0], tgt_coord[1], :, :].transpose()

    # Random vectors that obey these bounds (this is is R_i * alpha ^ i * w)
    obeying_y_comps = flt_to_int(np.random.uniform(low=bounds[:, 0],
                                                   high=bounds[:, 2]), rnd)
    obeying_x_comps = flt_to_int(np.random.uniform(low=bounds[:, 1],
                                                   high=bounds[:, 3]), rnd)
    obeying_vecs = np.stack((obeying_y_comps, obeying_x_comps), axis=-1)

    # Where these vectors will end up, assuming they started at tgt_coord
    final_locs = tgt_coord + obeying_vecs

    # Vectors to get from the current position to the final location
    # This is u_i in the paper (for every i)
    ui_s = final_locs - pos

    # Vectors to look at to minimize D
    vecs = np.append(ui_s, np.array([vec]), axis=0)

    # Now choose the best vector from vecs and the corresponding D-value
    best_vec, best_D = best_vector(vecs, pos, src_patches, trg_patches)

    return best_vec, best_D

# Input:
#   rect: Some (2+)-dimensional array
# Let:
#   v(x, y)  = (v_x, v_y) be a vector positioned at (x, y)
#   (t_x, t_y) = (x, y) + v(x, y)
#   B = maximum x position of rect
#   H = maximum y position of rect
# Output:
#   3D Array: Matrix where each element is a 4-vector in the following form (by element):
#     0: the lower bound for v_y such that t_y is in the range [0, H] and v_y doesn't exceed w
#     1: the lower bound for v_x such that t_x is in the range [0, B] and v_x doesn't exceed w
#     2: the upper bound for v_y such that t_y is in the range [0, H] and v_y doesn't exceed w
#     3: the upper bound for v_x such that t_y is in the range [0, B] and v_x doesn't exceed w
def make_vector_bounds(rect, r=None):

    # Dimensions of the rectangle
    H, B = edge_indices(rect)[2:]

    # Get the right value for r if it's not given to us
    if r is None:
        w_to_use = max(B, H) + 1
    else:
        w_to_use = r

    # Let:
    #   Vector at (x, y) = v(x, y) = (v_x, v_y)
    #   Base (max X) of the rectangle be B
    #   Height (max Y) of the rectangle be H
    # We note the following value restrictions:
    #   1 Position constraints (these are guaranteed):
    #       x in [0, B]
    #       y in [0, H]
    #   2 Vectors can't have lengths that exceed rectangle bounds:
    #       v_x in [-B, B]
    #       v_y in [-H, H]
    #   3 Vectors must end inside the rectangle:
    #       x + v_x in [0, B] --> v_x in [-x, B - x]
    #       y + v_y in [0, H] --> v_y in [-y, H - y]
    #   4 Vector component lengths can't exceed r (for random search)
    #       v_x in [-r, r]
    #       v_y in [-r, r]
    #
    # Assuming 1, we can make all 2,3,4 hold if we have the following:
    #   v_x in [max(-r, -x), min(r, B - x)]
    #   v_y in [max(-r, -y), min(r, H - y)]

    # Matrix of coordinates
    y_coords, x_coords = split_yx(coords_of(rect))

    # Lower bound matrices for v_x, v_y
    vx_lb = np.maximum(-w_to_use, -1 * x_coords)
    vy_lb = np.maximum(-w_to_use, -1 * y_coords)

    # Upper bound matrices for v_x, v_y
    vx_ub = np.minimum(w_to_use, B - x_coords)
    vy_ub = np.minimum(w_to_use, H - y_coords)

    # Stack the bound matrices together and return them
    return flt_to_int(np.dstack((vy_lb, vx_lb,
                                 vy_ub, vx_ub)),
                      rnd)

# Input:
#   rect: Some (2+)-dimensional array
#   alpha: the alpha in the paper
#   w: the w in the paper
# Output:
#   A 4D stack of vector-bounds arrays as described in make_vector_bounds.
#   For some layer i in the fourth dimension of this array, this layer is
#   equal to the vector-bound array with r = (w * alpha ** i)
def vec_bounds_stack(rect, alpha, w):

    # List of vector-bound arrays
    lst = []

    for i in range(num_iters_needed(w, alpha)):
        radius = w * (alpha ** i)
        lst += [make_vector_bounds(rect, r=radius)]

    # Stack the bounds arrays together on a new axis and return them
    ret_stack = np.stack(tuple(lst), axis=-1)

    return ret_stack
"""END OF FUNCTIONS FOR RANDOM SEARCH"""


"""FUNCTIONS THAT DEAL WITH FILLING IN THE NNF (PROPAGATION)"""
# Fills in the rest of a partial NNF created by partial_nnf
# The partial NNF must have it's top row and left column
# filled in
def fill_partial_nnf(partial_NNF, partial_D,
                     init_NNF, init_D,
                     bound_matr,
                     src_patches, trg_patches,
                     prop_enabled, rand_enabled):

    # NNF and D Matrices to return
    ret_NNF = partial_NNF.copy()
    ret_D = partial_D.copy()

    # Number of rows and columns in the NNF
    num_rows, num_cols = edge_indices(ret_NNF)[2:] + 1

    for row in range(1, num_rows):
        for col in range(1, num_cols):

            cur_pos = np.array([row, col])

            # Start off with initial NNF value
            best_vec = init_NNF[cur_pos[0], cur_pos[1]]

            if prop_enabled:

                # Vectors to compare: current, left, above
                # Snip these vectors so they don't go out of bounds
                above_vec = snip_vector(ret_NNF[row - 1, col],
                                        cur_pos, init_NNF)
                left_vec = snip_vector(ret_NNF[row, col - 1],
                                       cur_pos, init_NNF)

                # Now look for the best vector out of the three
                best_vec, best_D = best_vector(np.array([best_vec, above_vec, left_vec]),
                                               cur_pos, src_patches, trg_patches)

            if rand_enabled:

                best_vec, best_D = rand_search(
                    cur_pos,
                    best_vec,
                    bound_matr,
                    src_patches, trg_patches
                )

            # Update the values in the return matrix
            ret_NNF[row, col] = best_vec
            ret_D[row, col] = best_D

    return ret_NNF, ret_D

# "Fills in" the leftmost column and topmost row of a
# zeroed-out NNF to create a "fresh" NNF. "Filling in" is as follows:
#   1) Choose the better neighbour (we only have to choose from 1 neighbour when
#       filling out the extreme rows/columns)
#   2) Conduct a random search to find a better NN vector at the location
# Inputs:
#   init_NNF: the initial NNF to create a "fresh" NNF out of
#   init_D: the D-values of init_NNF
#   rand_matr: multi-dimensional array of bounds for vectors originating
#       at certain positions
#   src_patches: patches of the source image
#   trg_patches: patches of the target image
# Output:
#   A "fresh" NNF that can be propagated using the up-left rule
def partial_nnf(init_NNF, init_D,
                bound_matr,
                src_patches, trg_patches,
                prop_enabled, rand_enabled):

    # Prepare the matrices
    corner_only_NNF = np.zeros_like(init_NNF)
    corner_only_D = np.zeros_like(init_D)

    # Set the top-left corner to the initial NNF's top-left vector
    corner_only_NNF[0, 0] = init_NNF[0, 0]
    corner_only_D[0, 0] = init_D[0, 0]

    if rand_enabled:
        corner_only_NNF[0, 0], corner_only_D[0, 0] = rand_search(
            np.array([0, 0]),
            corner_only_NNF[0, 0],
            bound_matr,
            src_patches, trg_patches
        )

    # Fill in the left column
    lc_filled_NNF, lc_filled_D = fill_left_col(
        corner_only_NNF, corner_only_D,
        init_NNF, init_D,
        bound_matr,
        src_patches, trg_patches,
        prop_enabled, rand_enabled
    )

    # Fill in the top row and return the resultant matrix
    tr_lc_NNF, tr_lc_D =  fill_top_row(
        lc_filled_NNF, lc_filled_D,
        init_NNF, init_D,
        bound_matr,
        src_patches, trg_patches,
        prop_enabled, rand_enabled
    )

    return tr_lc_NNF, tr_lc_D

# Helper to prepare the left-most column of a "fresh" NNF
# Input:
#   fresh_NNF: the NNF to prepare the left-most column for
#   fresh_D: the D-values of fresh_NNF
#   init_NNF: the original NNF
#   init_D: the matrix of D-values of the NN vectors in init_NNF
#   bound_matr: multi-dimensional array of bounds for vectors originating
#       at certain positions
#   src_patches: patches of the source image
#   trg_patches: patches of the target image
#   prop_enabled: whether or not we should propagate
#   rand_enabled: whether or not we should do random search
# Output:
#   Modified fresh_NNF such that the leftmost column is "filled out"
def fill_left_col(fresh_NNF, fresh_D,
                  init_NNF, init_D, bound_matr,
                  src_patches, trg_patches,
                  prop_enabled, rand_enabled):

    # NNF and D Matrices to return
    ret_NNF = fresh_NNF.copy()
    ret_D = fresh_D.copy()

    # Number of rows in the NNF
    num_rows = edge_indices(ret_NNF)[2] + 1

    # Going down the left column of the fresh NNF
    for row in range(1, num_rows):

        # Our current position
        cur_pos = np.array([row, 0])

        # Start off with the initial NN vector
        best_vec = init_NNF[cur_pos[0], cur_pos[1]]

        if prop_enabled:
            # Try to improve it with the NN vector above
            # Snip it so it isn't out of bounds
            above_vec = snip_vector(ret_NNF[row - 1, 0],
                                    cur_pos, init_NNF)

            best_vec, best_d = best_vector(
                np.array([best_vec, above_vec]),
                cur_pos,
                src_patches, trg_patches
            )

        if rand_enabled:
            # Now conduct a random search around best_vec to improve it
            best_vec, best_d = rand_search(
                cur_pos,
                best_vec,
                bound_matr,
                src_patches, trg_patches
            )

        # Now fill in the current position in fresh_NNF with best_vec
        ret_NNF[cur_pos[0], cur_pos[1]] = best_vec
        ret_D[cur_pos[0], cur_pos[1]] = best_d

    # Return what we need
    return ret_NNF, ret_D

# Like fill_left_col, but fills out the top row instead
# in the same fashion
def fill_top_row(fresh_NNF, fresh_D,
                 init_NNF, init_D, bound_matr,
                 src_patches, trg_patches,
                 prop_enabled, rand_enabled):

    # NNF and D Matrices to return
    ret_NNF = fresh_NNF.copy()
    ret_D = fresh_D.copy()

    # Number of rows in the NNF
    num_cols = edge_indices(ret_NNF)[3] + 1

    # Going down the left column of the fresh NNF
    for col in range(1, num_cols):

        # Our current position
        cur_pos = np.array([0, col])

        # Start off with the initial NN vector
        best_vec = init_NNF[cur_pos[0], cur_pos[1]]

        if prop_enabled:
            # Try to improve it with the NN vector to left
            # Snip it so it's in bounds
            left_vec = snip_vector(ret_NNF[0, col - 1],
                                   cur_pos, init_NNF)
            best_vec, best_D = best_vector(
                np.array([best_vec, left_vec]),
                cur_pos,
                src_patches, trg_patches
            )

        if rand_enabled:
            # Now conduct a random search around best_vec to improve it
            best_vec, best_D = rand_search(
                cur_pos,
                best_vec,
                bound_matr,
                src_patches, trg_patches
            )

        # Now fill in the current position in fresh_NNF with best_vec
        # and the same place in ret_D's with the D-value
        ret_NNF[cur_pos[0], cur_pos[1]] = best_vec
        ret_D[cur_pos[0], cur_pos[1]] = best_D

    # Return what we need
    return ret_NNF, ret_D
"""END OF FUNCTIONS THAT DEAL WITH FILLING IN THE NNF (PROPAGATION)"""

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

    # Matrix such that element [x, y] = (x, y) + f(x, y)
    tgt_coords = coords_of(f) + f

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
