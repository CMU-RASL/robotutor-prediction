import helpers as hp
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

def compareFiles(video_filename):
    f = open('tempData/picture_side/' + video_filename[-6:-4] +'.txt')
    first = f.readlines()
    f.close()
    f = open('data/picture_sides/' + video_filename[-6:-4] +'_picture_sides.txt')
    second = f.readlines()
    f.close()
    second = second[1:] #get rid of first line
    if len(first) != len(second):
        print(len(first), len(second))
        return False
    for (x,y) in zip(first, second):
        if (x == 0 and y == "right") or (x == 1 and y == "left"):
            return False
    return True

def saveFrame(video_filename):
    cap = cv2.VideoCapture(video_filename) #video name
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        wind = frame[10:70,10:100,:]

        cv2.imwrite('templates/backbutton/not_pressed.png',wind)
        break
    cap.release()
    cv2.destroyAllWindows()


def test(video_filename):
    cap = cv2.VideoCapture(video_filename) #video name
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break
        wind = cv2.cvtColor(frame[10:70,10:100,:], cv2.COLOR_BGR2GRAY)
        output = wind.copy()
        # wind = cv2.cvtColor(frame[20:60,20:90,:], cv2.COLOR_BGR2GRAY)
        # plt.imshow(big)
        # plt.show()
        # break
        blur = cv2.GaussianBlur(wind,(3,3),0);
        circles = cv2.HoughCircles(wind,cv2.HOUGH_GRADIENT,1,20,
                                    param1=50,param2=22.5,minRadius=5,maxRadius=50)
        unpressed_image = cv2.imread('templates/backbutton/not_pressed.png',cv2.IMREAD_GRAYSCALE)
        ssim = structural_similarity(unpressed_image, wind)
        if (not circles is None and ssim > 0.88 and ssim < 0.96):
            radius = circles[0][0][2]
            if (radius < 6.9 or radius > 11):
                continue #restarts loop
            # plt.imshow(frame)
            # plt.show()
            print(circles)
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
        		# draw the circle in the output image, then draw a rectangle
        		# corresponding to the center of the circle
        	       cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        	       cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        	# show the output image
            cv2.imshow("output", np.hstack([wind, output]))
            cv2.waitKey(0)
            cv2.waitKey(1)
            print("found")
    cap.release()
    cv2.destroyAllWindows()

test('tempData/sample1.mp4')
# test('data/old_video/01.mp4')
# saveFrame('tempData/sample3.mp4')
