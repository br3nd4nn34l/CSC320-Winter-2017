import cv2 as cv, numpy as np


alpha_matte_path = """G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 2\\test_images\input-alpha.bmp"""
alpha_matte_ptr = cv.imread(alpha_matte_path, cv.IMREAD_GRAYSCALE)
boundaries = cv.findContours(alpha_matte_ptr, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)[1]
contour_img = np.zeros(alpha_matte_ptr.shape)
cv.drawContours(contour_img, boundaries, -1, (255,255,0),1)
cv.imwrite("Contour.tif", contour_img)