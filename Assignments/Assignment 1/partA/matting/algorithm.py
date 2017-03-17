## CSC320 Winter 2017 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv

# If you wish to import any additional modules
# or define other utility functions, 
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# Some constants I'm going to use for modularity

# Used when an operation is unsuccessful
unsuccessful = "unsuccessful"
# Used when an operation is successful
successful = "successful"
# Defining the color space that OpenCV uses in the arrays
G, B, R = 0, 1, 2

def look_for_none(lst):
    """([Object]) -> bool

    Returns whether or not there exists a None in the list. Uses the is keyword
    for comparisons rather than == to avoid a "future warning" that Python
    throws.

    :param lst: a list of python objects
    :return: whether or not there exists a None in lst
    """
    for x in lst:
        if x is None:
            return True
    return False

def stack_clones(numpy_arr, times_to_stack):
    """(np.array, int) -> numpy.ndarray

    Calls numpy.dstack on times_to_stack copies of numpy_arr on top of each
    other.


    >>> arr = np.array([[1, 2], [3,4]])
    >>> res = stack_clones(arr, 3)
    >>> exp = np.array([[[1, 1, 1],[2, 2, 2]],[[3, 3, 3],[4, 4, 4]]])
    >>> np.array_equal(exp, res)
    True

    """
    lst_to_stack = [numpy_arr for i in range(times_to_stack)]
    return np.dstack(tuple(lst_to_stack))

#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing 
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing 
    # algorithms. These images are initialized to None and populated/accessed by 
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = { 
            'backA': None, 
            'backB': None, 
            'compA': None, 
            'compB': None, 
            'colOut': None,
            'alphaOut': None, 
            'backIn': None, 
            'colIn': None, 
            'alphaIn': None, 
            'compOut': None, 
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self): 
        return {
            'backA':{'msg':'Image filename for Background A Color',
                     'default':None},
            'backB':{'msg':'Image filename for Background B Color',
                     'default':None},
            'compA':{'msg':'Image filename for Composite A Color',
                     'default':None},
            'compB':{'msg':'Image filename for Composite B Color',
                     'default':None},
        }

    # Same as above, but for the output arguments
    def mattingOutput(self): 
        return {
            'colOut':{'msg':'Image filename for Object Color',
                      'default':['color.tif']},
            'alphaOut':{'msg':'Image filename for Object Alpha',
                        'default':['alpha.tif']}
        }

    def compositingInput(self):
        return {
            'colIn':{'msg':'Image filename for Object Color',
                     'default':None},
            'alphaIn':{'msg':'Image filename for Object Alpha',
                       'default':None},
            'backIn':{'msg':'Image filename for Background Color',
                      'default':None},
        }

    def compositingOutput(self):
        return {
            'compOut':{'msg':'Image filename for Composite Color',
                       'default':['comp.tif']},
        }
    
    # Copy the output of the triangulation matting algorithm (i.e., the 
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and \
                (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the 
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    def _check_key_validity(self, keys):
        """([str]) -> (bool, [str])

		Checks the validity of keys in the _image directory. A key is
		valid if it points to a valid image.

		:return: a 2-tuple in the form:
			First: whether or not there exists a key in keys is invalid
			Second: list of invalid keys
		"""
        img_ptrs = map(lambda key: self._images[key],
                       keys)

        # One or more of the images is missing
        if look_for_none(img_ptrs):
            exists_invalid = True

            # Find which keys are missing (i.e. those that have null pointers)
            missing = filter(lambda key: self._images[key] is None,
                             keys)

        # All of the pointers point to images
        else:
            exists_invalid = False
            missing = []

        return (exists_invalid, missing)

    def _triangulationMattingHelper(self):
        # This function is only called after it has been verified that all of
        # the image pointers are valid (i.e. not None)

        # Abort if something goes wrong in the code inside this block
        try:
            # Note: from my experience on a 32-bit system, the performance benefit
            # of using any numpy type smaller than 32 bits is negligible when
            # compared with their 32 bit counterparts

            # Note that all the images stored in the system are matricies of
            # floats between 0 and 1

            # According to handout, this takes four inputs as follows
            # Two images of the foreground against the two backgrounds (f1, f2)
            f1 = self._images["compA"]
            f2 = self._images["compB"]

            # Two images of the two backgrounds (k1, k2)
            k1 = self._images["backA"]
            k2 = self._images["backB"]

            # According to Smith and Blinn's Matting paper, given:
            #   Backgrounds: k1, k2
            #   Foregrounds: f1, f2, where fi is an image of the foreground object on ki
            #   Colors Ci - color channel C in image i (pixel location is assumed
            #               to be the same across all images)
            # Let (for given color C = R|G|B):
            #   f_delta = Cf1 - Cf2
            #   k_delta = Ck1 - Ck2
            # Then:
            #   Numerator = Rf*Rk + Gf*Gk + Bf*Bk
            #   Denominator = Rk*Rk + Gk*Gk + Bk*Bk = Rk^2 + Gk^2 + Bk^2
            #   Complement = Numerator / Denominator
            #   Alpha = 1 - Complement

            # For numpy optimization, we will let Ci = vector(Ri, Gi, Bi)

            # Find the difference between the two backgrounds
            k_delta = k2 - k1
            # Find the difference between the two foregrounds
            f_delta = f2 - f1

            # Find each component of numerator in vector form
            numerator_vectors = k_delta * f_delta
            # Dot with (1, 1, 1) to sum vector components together
            # Interpret as float for proper division results later
            numerator = numerator_vectors.dot([1, 1, 1])

            # Find each component of denominator in vector form
            denominator_vectors = k_delta * k_delta
            # Dot with (1, 1, 1) to sum vector components together
            # Interpret as float for proper division results later
            denominator = denominator_vectors.dot([1, 1, 1])

            # Make a copy of the denominators, but with all the zeroes set to
            # 1, we'll replace the division result later with 1 anyways (this
            # is to avoid division errors)
            pseudo_denoms = denominator.copy()
            pseudo_denoms[pseudo_denoms == 0] = 1

            # Calculate the complement
            complement = (numerator / pseudo_denoms)
            # Set the pixels where denom == 0 to 1 (to emulate numpy pinv logic)
            complement[denominator == 0] = 0

            # Determine the alpha values by doing alpha = 1 - complement
            # Clip them so they don't violate the alpha in [0, 1] constraint
            alphas = np.clip(1 - complement, 0, 1)

            # Let the "alpha" dictionary pointer point to array of alphas
            self._images["alphaOut"] = alphas

            # Stack the alpha matrix on top of itself so it can be multiplied
            # pixel-wise with the foreground
            alpha_stacked = stack_clones(alphas, 3)

            # Since we determined a reasonable approximation of alpha using the
            # pseudo-inverse and matting equation says that:
            # C_n = C_f1 - C_k1 * (1 - alpha)
            # C_n = C_f2 - C_k2 * (1 - alpha)
            #   where
            #       C_n is the color of the target object
            #       C_fx is the color of the x-th composite image
            #       C_kx is the color of the x-th background image
            # Since we can't have C_n be equal to two different but approximately equal values, we'll take the
            # average between them for a better C_n value:
            # approx(C_n) = ((C_f1 - C_k1 * (1 - alpha)) + (C_f2 - C_k2 * (1 - alpha))) / 2
            #             = ((C_f1 + C_f2) - (1 - alpha) * (C_k1 + C_k2)) / 2

            bg_sum, fg_sum = k1 + k2, f1 + f2

            # These are the pixels that belong in the target object (e.g. if the imageset
            # is the flowers, the target is the flowers)
            fg_contrib = np.clip((fg_sum - ((1 - alpha_stacked) * bg_sum)) / 2, 0, 1)

            # Let the "colOut" dict pointer point to the target object pixels
            self._images["colOut"] = fg_contrib

            # Return true to indicate success here
            return True

        # If something does go wrong return False to indicate failure
        except Exception:
            return False

    def _createCompositeHelper(self):
        # This function is only called after it has been verified that all of
        # the necessary image pointers are valid (i.e. not None)

        # Try to run the code in this block
        try:

            # Get the foreground contribution
            fg_contrib = self._images["colIn"]

            # Get the matrix of alphas
            alphas = self._images["alphaIn"]

            # Determine the complement of the alpha that will be multiplied
            # with the new background
            alpha_complement = 1 - alphas

            # Load the new background
            new_back = self._images["backIn"]

            # Matting equation is as follows (for given color C = (R|G|B)):
            # C = Cf + ((alpha_complement) * Ck)
            # Where:
            #   C is the color of a pixel in the new composition
            #   Cf is the color of a pixel from the foreground's contribution
            #       (basically the original foreground * alpha)
            #   alpha_complement = 1 - alpha
            #   Ck is the color of the new background

            # Since colIn is assumed to already be the old foreground *
            # alpha, we don't have to do anything with it

            # Calculate the amount that the background will "contribute" to
            # the new composition
            background_contrib = new_back * alpha_complement

            # Use the matting equation to get the composition,
            composition = fg_contrib + background_contrib


            # Assign the dictionary pointer compOut to composition
            self._images["compOut"] = np.clip(composition, 0, 1)

            # If we've gotten to this point it means that everything prior to
            # this worked perfectly, so the operation must be successful
            return True

        # If the above code fails at any point, return False to indicate that
        # the operation failed
        except:
            return False

    #########################################
    # NOTE: ALL IMAGE VALUES WILL BE STORED AS FLOATS IN THE RANGE [0, 1]
    # THIS IS TO ENSURE THAT ALL CONVERSION IS ONLY DONE IN READING AND WRITING
    def readImage(self, fileName, key):
        """(str, str) -> (bool, str)

        Use OpenCV to read an image from a file and copy its contents to the
        matting instance's private dictionary object. The key
        specifies the image variable and should be one of the
        strings in lines 54-63. See run() in run.py for examples

        The routine should return True if it succeeded. If it did not, it should
        leave the matting instance's dictionary entry unaffected and return
        False, along with an error message

        """
        msg = "Opening {filename} was ".format(filename=fileName) + \
              "{status}"

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # Try to open the image (default is 3-channel color, which is what we
        # want to look at
        img_ptr = cv.imread(fileName)

        # If img_ptr is None, the operation failed
        if img_ptr is None:

            # Set message and success as needed
            success = False
            msg = msg.format(status=unsuccessful)

        # Otherwise the operation succeeded
        else:
            # Set message and success as needed
            success = True
            msg = msg.format(status=successful)

            # Divide by 255.0 to scale the intensities as floats between 0 and 1
            scaled_img = img_ptr.astype(np.float32) / 255

            # Get the right key in the dictionary to point to the image's
            # file pointer
            self._images[key] = scaled_img

        #########################################
        return success, msg

    def writeImage(self, fileName, key):
        """(str, str) -> (bool, str)

        Use OpenCV to write to a file an image that is contained in the
        instance's private dictionary. The key specifies the which image
        should be written and should be one of the strings in lines 54-63.
        See run() in run.py for usage examples

        The routine should return True if it succeeded.
        If it did not, it should return False, along with an error message

        """

        # Make the message template for future use
        msg = "Writing {key} to {filename} was".format(filename=fileName,
                                                       key=key) + \
              " {status}"

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # Assigning image pointer to name
        img_ptr = self._images[key]

        # If the image doesn't exist, there's no way we can write it to
        # anything
        if img_ptr is None:
            msg = msg.format(status=unsuccessful)
            success = False

        # If it isn't None we might be able to still write it to a file
        else:

            # According to my specification, the image is a bunch of
            # floats between [0, 1] that need to be scaled up to [0, 255]
            # So, multiply by 255, round it and then interpret as a uint8
            scaled_intensities = np.round(img_ptr * 255).astype(np.uint8)

            # Try to save the image
            write_attempt = cv.imwrite(fileName, scaled_intensities)

            # If write_attempt is False it didn't work
            if not write_attempt:
                msg = msg.format(status=unsuccessful)
                success = False

            # If write_attempt is True, it worked
            else:
                msg = msg.format(status=successful)
                success = True

        #########################################
        return success, msg


    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # object.
    def triangulationMatting(self):
        """
        success, errorMessage = triangulationMatting(self)
        
        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Triangulation matting was {status}.'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # According to handout, this takes the following four inputs:
        # compA, compB, backA, backB

        # List the pointers that we need to look at
        keys = ["compA", "compB", "backA", "backB"]

        # Use the key checking method to see:
        #   (If this operation will fail,
        #   What keys are missing)
        (exists_invalid, missing) = self._check_key_validity(keys)

        # One or more of the images is missing
        if exists_invalid:

            # Obviously this operation will be unsuccessful
            success = False
            msg = msg.format(status=unsuccessful)

            # Inform the user about which images are missing
            msg += " The following are not associated to any images: " \
                   "{missed}.".format(missed=", ".join(missing))

        # No images are missing
        else:
            result = self._triangulationMattingHelper()
            if result:
                msg = msg.format(status=successful)
            else:
                msg = msg.format(status=unsuccessful)
                msg += " Something went wrong while manipulating {imgs} to " \
                       "obtain alpha values and foreground contribution.".format(
                    imgs = ", ".join(keys)
                )
            success = result

        #########################################

        return success, msg

    def createComposite(self):
        """
        success, errorMessage = createComposite(self)
        
        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Compositing was {status}.'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################

        # List the pointers that we need to look at
        keys = ["alphaIn", "colIn", "backIn"]

        # Use the key checking method to see:
        #   (If this operation will fail,
        #   What keys are missing)
        (exists_invalid, missing) = self._check_key_validity(keys)

        # One or more of the images is missing
        if exists_invalid:
            # Obviously this operation will be unsuccessful
            success = False
            msg = msg.format(status=unsuccessful)

            # Inform the user about which images are missing
            msg += " The following are not associated to any images: " \
                   "{missed}.".format(missed=", ".join(missing))
        else:
            result = self._createCompositeHelper()
            if result:
                msg = msg.format(status=successful)
            else:
                msg = msg.format(status=unsuccessful)
                msg += " Something went wrong while manipulating {imgs} to " \
                       "obtain the composite.".format(
                    imgs=", ".join(keys)
                )
            success = result

        #########################################

        return success, msg
