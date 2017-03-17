import numpy as np, cv2 as cv, pprint

pp = pprint.PrettyPrinter(indent=4)

k1_path = """G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Back\A.jpg"""
k2_path = """G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Back\B.jpg"""
f1_path = """G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Comp\A.jpg"""
f2_path = """G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Comp\B.jpg"""

ref_alpha_path = """G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Large\Flowers\Results\Ref\(compA, compB) = (A, B) Alpha.tif"""
my_alpha_path = """G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Large\Flowers\Results\Mine\(compA, compB) = (A, B) Alpha.tif"""

ref_alpha_ptr = cv.imread(ref_alpha_path).astype(np.int32)
my_alpha_ptr = cv.imread(my_alpha_path).astype(np.int32)

k1_ptr = cv.imread(k1_path).astype(np.int32)
k2_ptr = cv.imread(k2_path).astype(np.int32)
f1_ptr = cv.imread(f1_path).astype(np.int32)
f2_ptr = cv.imread(f2_path).astype(np.int32)

back_diff = k1_ptr - k2_ptr
fore_diff = f1_ptr - f2_ptr

alpha_diff = my_alpha_ptr - ref_alpha_ptr

white = [255, 255, 255]
red = [0, 0, 255]
blue = [255, 0, 0]
green = [255, 0, 0]




marked_errors = np.absolute(np.copy(alpha_diff))
print marked_errors.shape
dims = marked_errors.shape
marked_errors[0:dims[0], 0:dims[1]] = white
marked_errors[np.where((alpha_diff == [0, 0, 0]).all(axis = 2))] = white
marked_errors[np.where((alpha_diff == [1, 1, 1]).all(axis = 2))] = blue
marked_errors[np.where((back_diff == [0, 0, 0]).all(axis = 2))] = green

cv.imwrite("Alpha Diff Markers.tif", marked_errors * 5)

alpha_diff_flat = alpha_diff.flatten()


(num, freq) = np.unique(alpha_diff_flat, return_counts=True)
alpha_diff_freq_dict = {}
for i in range(len(num)):
    cur_num, cur_freq = num[i], freq[i]
    alpha_diff_freq_dict[cur_num] = cur_freq
print "Number of Pixels: {num}".format(num=alpha_diff_flat.size)
print "Number of Errors: {num}".format(num=alpha_diff_flat.size - (alpha_diff_freq_dict[0]))
print "Alpha Error frequency:"
pp.pprint(alpha_diff_freq_dict)