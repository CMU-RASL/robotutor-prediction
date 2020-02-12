import os
import cv2
import numpy as np
from datetime import datetime, timedelta

def gray_screen(video_filename):

    cap = cv2.VideoCapture('data/video/'+video_filename)

    sides = []
    #is this the length of the video?
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    seenGray = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = []
    end = []
    num = 0

    while(cap.isOpened()):

        #if num > 12:
        #    break

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        height, width, channels = frame.shape
        snap = frame[height//2 - 50 : height//2 + 50,
                          width // 2 - 100 : width // 2 + 100,:]

        colors, count = np.unique(snap.reshape(-1,snap.shape[-1]),
                                  axis=0, return_counts=True)

        if [244,249,255] in colors and len(colors) == 1:
            print(colors)
            cv2.imwrite('data/activity_time/feedbackFrame.jpg', frame)

        if not seenGray and len(colors) == 1 and np.array_equal(colors[0], [136,136,136]) :
            seenGray = True
            start.append(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
            #print("start", cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
            #print("start", cap.get(cv2.CAP_PROP_POS_MSEC))
            #cv2.imwrite('data/activity_time/frame1.jpg', frame)

        elif seenGray == True and len(colors) != 1:
            seenGray = False
            end.append(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
            cv2.imwrite('data/activity_time/frame' + str(num) +'.jpg', frame)
            print("end", cap.get(cv2.CAP_PROP_POS_MSEC))
            num += 1
    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()
    f = open('data/activity_time/' +
         video_filename[:2]+'activity.txt', 'w+')
    for i in range(max(len(start), len(end))):
        if i < len(start):
            f.write("start: " + str(start[i]) + "\n")
        if i < len(end):
            f.write("end: " + str(end[i]) + "\n")
    f.close()

def main():
    print ("start")
    video_filenames = os.listdir('data/video')
    gray_screen("01.mp4")

main()
