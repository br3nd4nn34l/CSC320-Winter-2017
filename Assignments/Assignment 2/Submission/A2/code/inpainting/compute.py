## CSC320 Winter 2017 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

# For convolution
from scipy import signal

# Constants for modularity
# For boolean masks
off = 0
on = 255

# For polynomial fitting
poly_deg = 3

# Sobel 3x3 Kernels
sx_3 = np.array([[1, 0, -1],
                 [2, 0, -2],
                 [1, 0, -1]],
                dtype=float)
sy_3 = sx_3.transpose()

# Scharr 3x3 Kernels
scharr_x = np.array([[3, 0, -3],
                     [10, 0, -10],
                     [3, 0, -3]])
scharr_y = scharr_x.transpose()

# Gaussian 3x3 Blur Kernel
g3 = np.array([[1, 2, 1],
               [2, 4, 2],
               [1, 2, 1]],
              dtype=float)

# For storing Sobel kernels to save on calculation time
# Initialize with 3x3 Sobel kernels
sobel_dict = {(3, 0):sx_3,
              (3, 1):sy_3}

# Function that generates an NxN Sobel Kernel
# Based on recursive answer to following stack exchange thread:
# http://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size
# Axis mapping: (0, 1) <-> (x, y)
# N must be odd!
def sobel_n(N, axis):

    # Grab the kernel if it already exists
    if (N, axis) in sobel_dict:
        return sobel_dict[(N, axis)]

    # Save compute time in the event that we can transpose an existing kernel
    elif (N, 1 - axis) in sobel_dict:
        sobel_dict[(N, axis)] = sobel_dict[(N, 1 - axis)].transpose()
        return sobel_dict[(N, axis)]

    # Have to actually calculate the kernel
    else:
        result = signal.convolve(g3, sobel_n(N - 2, axis))
        sobel_dict[(N, axis)] = result
        return result

# Function that convolves the central element of an NxN array with an NxN kernel
# Note that the convolution of the central element has to be the dot product of
# the two flattened matrices
def central_convolve(arr, kernel):
    return np.dot(arr.flatten(), kernel.flatten())

# Wrapper function for getting the pixel values from an image
# inside the area of patch. Kwarg for radius allows for custom
# radius to be chosen when needed.
def patch_vals_from_image(image, patch, radius=None):
    if radius is None:
        radius = patch.radius()
    patch_coords = (patch.row(), patch.col())  # coordinates of patch center
    return copyutils.getWindow(image, patch_coords, radius)[0] # first element is the actual values

# Function that returns a boolean mask in the following form:
# 1 where a pixel resides inside image
# 0 where a pixel is outside image
def containment_mask(image, patch, radius=None):
    if radius is None:
        radius = patch.radius()
    patch_coords = (patch.row(), patch.col())  # coordinates of patch center
    bool_matrix = copyutils.getWindow(image, patch_coords, radius)[1]  # second element is literally what we want

    # Convert (True, False) -> (1, 0)
    ret_matrix = np.zeros_like(bool_matrix, dtype=int)
    ret_matrix[bool_matrix] = 1

    return ret_matrix

# Function for stripping off the border values of a 2D array.
# Input array is assumed to be square, with odd side length
# Example
#   Input: [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]]
#   Output: [[5]]
def strip_borders(array_2d):
    if (len(array_2d.shape) != 2) or \
            (array_2d.shape[0] != array_2d.shape[1]):
        return None
    else:
        max_index = array_2d.shape[0] - 1
        return array_2d[1:max_index, 1:max_index]

# Function that returns the side length of a patch
def side_length(patch):
    return (2 * patch.radius()) + 1

# Function that returns the area of a patch (inside the image)
def area(patch, image):
    patch_coords = (patch.row(), patch.col())
    inside_img = copyutils.getWindow(image, patch_coords, patch.radius())[1]
    return np.count_nonzero(inside_img)

# Function that returns the the most precise kernel possible
# for derivative approximations, given a kernel size.
# Axis (0, 1) <-> (x, y)
def best_deriv_kernel(kernel_size, axis):
    if kernel_size == 3:
        if axis == 0:
            return scharr_x
        else:
            return scharr_y
    else:
        return sobel_n(kernel_size, axis)

# Function that computes the normal to a 2D vector
def normal_to(vector):
    x_comp = -1 * vector[1]
    y_comp = vector[0]
    return np.array([x_comp, y_comp])

# Function that normalizes a vector to magnitude 1
def unit_vector(vector):
    vec_float = vector.astype(float)
    mag = np.sum(np.dot(vec_float, vec_float)) ** 0.5
    if mag > 0:
        return vec_float / mag
    else:
        return vector

##########################################

#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#
def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Getting the pixel values from the different images INSIDE THE CURRENT PATCH
    # Scaling between 0 and 1
    conf_vals = patch_vals_from_image(confidenceImage, psiHatP).astype(float) / on
    filled_vals = patch_vals_from_image(filledImage, psiHatP).astype(float) / on # I - Omega in paper

    # Numerator: Sum of all confidence values inside patch AND inside(I - Omega)
    nume = np.sum(conf_vals * filled_vals) # if the pixel is IN (I - Omega) it will be added, otherwise ignored

    # Denominator: area of the patch (inside the image)
    deno = area(psiHatP, filledImage)

    # Scale result between 0 and 255
    result = (nume / deno) * 255

    # Round the result and convert to uint8
    result = np.round(result).astype(np.uint8)

    #########################################

    return result

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################


    # Define Boolean Mask of predicate P = mask:
    # Valued as follows (for coordinates (x, y)):
    # If pixel at (X, Y) satisfies P, mask[X, Y] = 1
    # Otherwise, mask[X, Y] = 0

    # Method to find delta_I_q:
    # 1) Find delta_I for all pixels in patch psiHatP (using Kernel convolutions)
    # 2) Find boolean masks for the following predicates:
    #       Note: do not need to cover (q in psiHatP), as step 1 already does this
    #       a) (q has been filled)
    #       b) (q is inside the borders of the image)
    # 3) Do element-wise multiplication between all masks to compute a
    #       boolean mask for the intersection/and of all predicates
    # 4) Do element-wise multiplication between and-mask and delta_I to find all valid delta_I's
    # 5) Compute the magnitude of each valid delta_I
    # 6) Return the values for the delta_I with the greatest magnitude

    # Step 1: Find delta_I for all pixels in patch (will be a matrix of 2D-vectors)

    # Get the window that we can use for Sobel
    # If we are using the classic 3x3 Sobel kernel to convolve the image,
    # we need the width of this window to be 1 more than the patch (so add 1 to radius)
    color_patch = patch_vals_from_image(inpaintedImage, psiHatP,
                                        radius=(psiHatP.radius() + 1)).astype(float)
    # Average the color values to get the grayscale version
    gray_scale_floats = np.dot(color_patch, np.array([1, 1, 1])) / 3

    # Re-interpret as an int
    gray_scale_ints = np.round(gray_scale_floats).astype(np.uint8)

    # Use Scharr to find approximations of x, y derivatives (for the patch)
    # These will be the same size as the larger patch
    # Applying Scharr (more accurate than Sobel for kernel size of 3)
    I_x_big = cv.Scharr(gray_scale_ints, cv.CV_64F, 1, 0) # First derivative WRT x
    I_y_big = cv.Scharr(gray_scale_ints, cv.CV_64F, 0, 1) # First derivative WRT y

    # Strip off the border pixels from the I_x_big, I_y_big - they won't be valid
    I_x = strip_borders(I_x_big)
    I_y = strip_borders(I_y_big)
    # Make matrix where each element is [I_x, I_y] (depth-wise stack)
    all_delta_I = np.dstack((I_x, I_y))

    # Step 2: Find boolean masks
    # Pixels residing within filled region
    filled_patch = patch_vals_from_image(filledImage, psiHatP).astype(int) # 255 if in filled region, 0 otherwise
    filled_mask = filled_patch.copy() // on # Turn the 255's into 1's
    # Pixels within the borders of the image
    within_border_mask = containment_mask(inpaintedImage, psiHatP).astype(int)

    # Step 3: Find logical-and mask through multiplication
    and_mask = filled_mask * within_border_mask

    # Stack and-mask depth-wise so it can be multiplied element-wise with matrix of deltaI
    and_mask = np.dstack((and_mask, and_mask))

    # Step 4: Find valid delta_I's using element-wise multiplication (invalids get set to 0)
    valid_delta_I = all_delta_I * and_mask

    # Step 5: find magnitude of each valid delta_I
    # Can ignore square-root operation (if function f(x) is monotonically increasing,
    # f(g(x)) is maxed wherever g(x) is maxed)
    mags = (valid_delta_I * valid_delta_I).dot([1, 1])

    # Step 6: Pick the delta_I with the largest magnitude
    # Coordinates of the delta_I with the largest magnitude
    coords_of_largest = np.unravel_index(np.argmax(mags), mags.shape)
    # Vector at the largest mag's coordinates
    target_vec = valid_delta_I[coords_of_largest]

    # Assigning derivative vectors to the right component in target
    Dx, Dy = target_vec

    #########################################
    
    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#
def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Note: I won't be using the fill-front to determine the normal,
    # just the fill-region

    # Method to find the normal to the fill-front:
    # 1) Figure out which pixels of the fill-region belong inside the current patch
    # 2) Find the rates of change with respect to X and Y at the central pixel of the group
    #       obtained in (1) (use Kernel convolution approximations)
    # 3) These rates of change can be combined into the gradient vector of the fill-front
    #       deltaF = (F_x, F_y).
    # 4) Return the result after normalizing deltaF

    # Step 1: determine which pixels of the fill-region belong inside the current patch
    patch_fill_front = patch_vals_from_image(fillFront, psiHatP) # Fill front pixels within patch (normalize to 1)

    # Edge Case: fill front is only one pixel (degenerate) - return None, None
    if np.count_nonzero(patch_fill_front) == 1:
        Nx = None
        Ny = None

    # Otherwise fill front is well-defined, so get to work
    else:

        # Step 2: find rates of change with respect to X and Y (using kernel convolutions)
        # Just convolve the central pixel with the largest/best possible derivative approximation
        # matrix (that can fit over the patch)

        # Find all filled pixels in the current window
        patch_filled = patch_vals_from_image(filledImage, psiHatP).astype(float)

        # Find kernel that we can use with most accuracy
        width = side_length(psiHatP)
        sobel_x = best_deriv_kernel(width, 0)
        sobel_y = best_deriv_kernel(width, 1)

        # Convolve the selection with Scharr or Sobel (whichever is more accurate for the kernel size)
        # Only take the value at the central pixel
        F_x = central_convolve(patch_filled, sobel_x) # First derivative WRT x
        F_y = central_convolve(patch_filled, sobel_y) # First derivative WRT y

        # Step 3: compute delta F and unit-vector it
        delta_F = unit_vector(np.array([F_x, F_y]))

        # Step 4: set return components to correct vector components
        Nx, Ny = delta_F

    return Ny, Nx

