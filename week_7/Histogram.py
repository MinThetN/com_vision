import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
# Load image in grayscale
img = cv.imread("../week_6/images/cat.jpg", cv.IMREAD_GRAYSCALE)
assert img is not None, "Image not found!"
 
# 1. Histogram using OpenCV
hist_cv = cv.calcHist([img], [0], None, [256], [0, 256])
 
# 2. Histogram using NumPy
hist_np, bins_np = np.histogram(img.ravel(), 256, [0, 256])
 
# 3. Faster NumPy method
hist_fast = np.bincount(img.ravel(), minlength=256)
 
# Plot all three for comparison
plt.figure(figsize=(10, 6))
plt.plot(hist_cv, label='OpenCV - calcHist', color='blue')
plt.plot(hist_np, label='NumPy - histogram', color='green', linestyle='dashed')
plt.plot(hist_fast, label='NumPy - bincount', color='red', linestyle='dotted')
plt.legend()
plt.title('Grayscale Histogram Comparison')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.grid(True)
plt.xlim([0, 256])
plt.show()
 
# Color image histogram

img_color = cv.imread("../week_6/images/cat.jpg")
assert img_color is not None, "Image not found!"
 
color = ('b', 'g', 'r')
plt.figure(figsize=(8, 4))
for i, col in enumerate(color):
    histr = cv.calcHist([img_color], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.xlim([0, 256])
plt.grid(True)
plt.show()
 

# Masked region histogram

mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv.bitwise_and(img, img, mask=mask)
 
hist_masked = cv.calcHist([img], [0], mask, [256], [0, 256])
 
# Plot full vs masked histogram
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(2, 2, 2), plt.imshow(mask, cmap='gray'), plt.title('Mask')
plt.subplot(2, 2, 3), plt.imshow(masked_img, cmap='gray'), plt.title('Masked Image')
plt.subplot(2, 2, 4), plt.plot(hist_cv, label='Full Image')
plt.plot(hist_masked, label='Masked Region')
plt.legend(), plt.title('Histogram Comparison'), plt.xlim([0, 256])
plt.tight_layout()
plt.show()