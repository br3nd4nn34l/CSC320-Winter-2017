import cv2 as cv, numpy as np

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
        return np.average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr

def scale_img(img):
	return img.astype(np.float) * (255 / img.max())


def extract_colors(img):
    img = img.astype(np.float)
    sqr_norms = img * img
    norms = sqr_norms.dot([1, 1, 1]) ** 0.5
    stacked_norms = np.stack((norms, norms, norms), 2)
    unit_colors = img / stacked_norms
    return unit_colors * 255

def print_avg_intensities(img, img_name):
    if len(img.shape) == 2: # Grayscale
        print "Average Intensity of {img_name}: {avg}".format(img_name=img_name,
                                                              avg=np.average(img))
    elif len(img.shape) == 3:  # BGR Image
        print "Average Intensities of {img_name}: R: {r_avg}, " \
              "G: {g_avg}, " \
              "B: {b_avg}".format(img_name=img_name,
                                  b_avg=np.average(img[:, :, 0]),
                                  g_avg=np.average(img[:, :, 1]),
                                  r_avg=np.average(img[:, :, 2]))


k1 = cv.imread("""G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Back\A.jpg""").astype(float)
k2 = cv.imread("""G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Back\B.jpg""").astype(float)

f1 = cv.imread("""G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Comp\A.jpg""").astype(float)
f2 = cv.imread("""G:\Computer Science Work\CSC320-Winter-2017\Assignments\Assignment 1\TestImages\Small\Flowers\Comp\B.jpg""").astype(float)


shade_independent = None

k1_gray = to_grayscale(k1)
k2_gray = to_grayscale(k2)
f1_gray = to_grayscale(f1)
f2_gray = to_grayscale(f2)

k1_colors = extract_colors(k1)
k2_colors = extract_colors(k2)
f1_colors = extract_colors(f1)
f2_colors = extract_colors(f2)

k_color_diff = np.abs(k1_colors - k2_colors)

print_avg_intensities(k1, "Backing A")
print_avg_intensities(k2, "Backing B")
print_avg_intensities(k1_colors, "Backing A (pure color)")
print_avg_intensities(k2_colors, "Backing B (pure color)")

print_avg_intensities(k_color_diff, "|Backing B (pure color) - Backing A (pure color)|")

diff = k1 - k2

print np.average(k2)

cv.imwrite("k1 colors.tif", k1_colors)
cv.imwrite("k2 colors.tif", k2_colors)

cv.imwrite("K Color Diff.tif", k_color_diff)
cv.imwrite("K Color Diff Grayscale.tif", to_grayscale(k_color_diff))
