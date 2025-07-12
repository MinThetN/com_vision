import cv2
import numpy as np
import os

image_path = os.path.join("images/star.jpg")  # Make sure 'images/shapes.png' exists
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert to grayscale and apply threshold
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the image
contour_img = image.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Print contour features
for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0

# Display images
cv2.imshow("Original Image", image)
cv2.imshow("Threshold", thresh)
cv2.imshow("Contours", contour_img)

print("Press any key to close all windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()
