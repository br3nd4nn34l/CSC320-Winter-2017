import sys

import cv2
from scipy.linalg import norm
from scipy import sum, average

def distances(img1_ptr, img2_ptr):
    # read images as 2D arrays (convert to grayscale for simplicity)
    img1_gray = to_grayscale(img1_ptr).astype(float)
    img2_gray = to_grayscale(img2_ptr).astype(float)
    # compare
    n_m, n_0 = compare_images(img1_gray, img2_gray)
    print "Manhattan norm:", n_m, "/ per pixel:", n_m / img1_ptr.size
    print "Zero norm:", n_0, "/ per pixel:", n_0*1.0 / img1_ptr.size

def compare_images(img1, img2):
    # normalize to compensate for exposure difference
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

if __name__ == "__main__":
    main()