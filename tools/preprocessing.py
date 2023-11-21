import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform

def crop_box(image_path):
    src = cv2.imread(image_path)
    dst = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10, 
                                      blockSize=3, 
                                      useHarrisDetector=True, k=0.03)
    corners = corners.astype(int)
    canny = cv2.Canny(src,100,300)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(canny, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)
    contours, _ = cv2.findContours(imgThreshold,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key = cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    box = np.where(box<0,0,box)
    cv2.drawContours(dst,[box],-1,(255,0,0),20)
    transform_image = four_point_transform(src, box.reshape(4, 2))
    transform_image = cv2.fastNlMeansDenoisingColored(transform_image,None,10,10,7,21)
    return src, dst, transform_image