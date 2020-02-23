import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def feedBack(yellow,green,red):
    yc, ycount = np.unique(yellow.reshape(-1,yellow.shape[-1]),
                          axis=0, return_counts=True)
    gc, gcount = np.unique(green.reshape(-1,green.shape[-1]),
                          axis=0, return_counts=True)
    rc, rcount = np.unique(red.reshape(-1,red.shape[-1]),
                          axis=0, return_counts=True)
    if np.array_equal(yc[ycount.argmax()], [50,236,250]) and np.array_equal(gc[gcount.argmax()], [68,228,55]) and np.array_equal(rc[rcount.argmax()], [55,54,232]):
        return True
    return False

def feedBackType(frame):
    height, width, channels = frame.shape
    yellow = frame[height // 2 - 110 : height // 2 - 100, width // 4 - 25 : width // 4 + 25]
    green = frame[height // 5 + 65 : height // 5 + 75, width // 4 - 25 : width // 4 + 25]
    red = frame[4*height // 5 - 85 : 4*height // 5 - 75, width // 4 - 25 : width // 4 + 25]

    yc, ycount = np.unique(yellow.reshape(-1,yellow.shape[-1]), axis=0, return_counts=True)
    gc, gcount = np.unique(green.reshape(-1,green.shape[-1]), axis=0, return_counts=True)
    rc, rcount = np.unique(red.reshape(-1,red.shape[-1]), axis=0, return_counts=True)

    g= gc[gcount.argmax()]
    y = yc[ycount.argmax()]
    r = rc[rcount.argmax()]

    if (np.array_equal(g, [208, 246, 208])):
        return "green"
    elif (np.array_equal(y, [207, 244, 248])):
        return "yellow"
    elif (np.array_equal(r, [207, 208, 247])):
        return "red"
    return None


def gray_screen(video_filename):

    cap = cv2.VideoCapture('data/video/01.mp4') #video name
    prevFrame = None

    sides = []
    #is this the length of the video?
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seenGray = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = []
    end = []
    num = 1

    feedBack_num = 1
    seenfeedBack = False

    while(cap.isOpened()):

        #if num > 12:
        #    break

        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        height, width, channels = frame.shape
        yellow = frame[height // 2 - 10 : height // 2 + 10, width // 4 - 25 : width // 4 + 25]
        green = frame[height // 5 - 10 : height // 5 + 10, width // 4 - 25 : width // 4 + 25]
        red = frame[4*height // 5 - 10 : 4*height // 5 + 10, width // 4 - 25 : width // 4 + 25]

        snap = frame[height//2 - 50 : height//2 + 50,
                          width // 2 - 100 : width // 2 + 100,:]


        if feedBack(yellow, green, red):
            f = feedBackType(frame)
            if not seenfeedBack and f != None:
                print(f, feedBack_num)
                cv2.imwrite('tempData/activity_time/feedback'+str(feedBack_num)+'.jpg', frame)
                feedBack_num += 1
                seenfeedBack = True
        else:
            seenfeedBack = False


        colors, count = np.unique(snap.reshape(-1,snap.shape[-1]),
                                  axis=0, return_counts=True)

        if not seenGray and len(colors) == 1 and np.array_equal(colors[0], [136,136,136]) :
            seenGray = True
            start.append(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
            #print("start", cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
            #print("start", cap.get(cv2.CAP_PROP_POS_MSEC))
            #cv2.imwrite('data/activity_time/frame1.jpg', frame)

        elif seenGray == True and len(colors) != 1:
            seenGray = False
            end.append(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
            cv2.imwrite('tempData/activity_time/frame' + str(num) +'.jpg', frame)
            print("end", cap.get(cv2.CAP_PROP_POS_MSEC))
            num += 1
    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()
    '''
    f = open('tempData/activity_time/' +
         video_filename[:2]+'activity.txt', 'w+')
    for i in range(max(len(start), len(end))):
        if i < len(start):
            f.write("start: " + str(start[i]) + "\n")
        if i < len(end):
            f.write("end: " + str(end[i]) + "\n")
    f.close()
    '''
def main():
    video_filenames = os.listdir('data/video')
    gray_screen("01.mp4")

main()
