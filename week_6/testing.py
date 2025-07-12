import cv2 as cv

img = cv.imread("images/cat.jpg")

cv.imshow("Cat", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

canny = cv.Canny(img, 125, 175)
cv.imshow("Canny Edges", canny)


cv.waitKey(10000)
cv.destroyAllWindows()



