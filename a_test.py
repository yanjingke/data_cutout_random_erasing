from random_eraser import get_random_eraser
from PIL import Image
import cv2 as cv
img="00c0d8bbab5d4a3aa16d51beffca5f93.jpg"
image=cv.imread(img)
print(image.shape)
eraser = get_random_eraser()
a=eraser(image)
# get_random_eraser()
# print(img2)
cv.imshow("aa",a)
cv.waitKey(0)
# img.show()