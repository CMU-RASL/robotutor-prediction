import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import helpers as hp

def get_picture_side(video_filename):
    cap = cv2.VideoCapture(video_filename) #video name
    f = open('tempData/picture_side/' + video_filename[-6:-4] +'.txt', 'w+')
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        f.write(str(hp.get_picture_side(frame))+"\n")
    cap.release()
    cv2.destroyAllWindows()
    f.close()


def feedBackScreen(yellow,green,red):
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
        return 1
    elif (np.array_equal(y, [207, 244, 248])):
        return 0
    elif (np.array_equal(r, [207, 208, 247])):
        return -1
    return None

def generateData(video_filename):
    cap = cv2.VideoCapture(video_filename) #video name

    f = open('tempData/picture_side/'+video_filename[-6:-4]+'.txt')
    all_sides = f.readlines()
    f.close()

    headers = ""
    all_openface = []
    with open('data/openface/' + video_filename[-6:-4]+'_crop.csv', mode='r') as csv_file:
           csv_reader = csv.reader(csv_file, delimiter=',')
           for ii, line in enumerate(csv_reader):
               if ii == 0:
                   headers = [s.strip() for s in line]
               else:
                   all_openface.append(hp.get_openface_features(headers, line))

    #is this the length of the video?
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seenGray = False
    fps = cap.get(cv2.CAP_PROP_FPS)
    start = []
    end = []
    num = 1

    feedBack_num = 0
    seenfeedBack = False
    allfeedBack = []
    activity_start = False #true means activity has started ans false means ended
    frame_count = 0
    totalframe_count = 0 #use this to extract precomputed data
    activity_name = ""
    times = []
    openface = []
    picture_sides = []
    lines = []
    carry = 0
    record = False #not all screens useful

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        totalframe_count += 1

        if activity_name == "":
            activity = hp.get_activity_type(frame)
            if activity != 'n/a' and not activity_start:
                activity_start = True
                activity_name = activity
        else:
            if activity_start: #activity starting
                print("activity started")
                record = True
                #reset all variables
                frame_count = 0
                times = []
                picture_sides = []
                carry = 0
                lines = []
                openface = []
            activity_start = False
        if not record: continue

        frame_count += 1
        times.append(frame_count/fps)
        openface.append(all_openface[totalframe_count-1])
        picture_sides.append(all_sides[totalframe_count-1])
        frac = hp.get_read_fraction(frame, picture_sides[len(picture_sides)-1])
        lines.append(frac+carry)
        if frac == 1:
            carry += 1

        # if activity_start == False: #activity ended
        #     createFeatures(video_filename, activityname_name, feedBack_num, allfeedBack[len(allfeedBack)-1], times)

        # height, width, channels = frame.shape
        # yellow = frame[height // 2 - 10 : height // 2 + 10, width // 4 - 25 : width // 4 + 25]
        # green = frame[height // 5 - 10 : height // 5 + 10, width // 4 - 25 : width // 4 + 25]
        # red = frame[4*height // 5 - 10 : 4*height // 5 + 10, width // 4 - 25 : width // 4 + 25]

        # snap = frame[height//2 - 50 : height//2 + 50,
        #                   width // 2 - 100 : width // 2 + 100,:]

        fb = feedBackType(frame)
        if fb != None and not seenfeedBack:
            if fb != None:
                print("feedback", fb)
                allfeedBack.append(fb)
                feedBack_num += 1
                seenfeedBack = True
                #cv2.imwrite('tempData/activity_time/feedBack'+str(feedBack_num)+'.jpg', frame)
                createFeatures(video_filename, activity_name, feedBack_num, fb, times, openface, lines, picture_sides)
                record = False
                activity_name = ""
        elif fb == None:
            seenfeedBack = False



        '''
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
        '''
    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()
    print(totalframe_count)

def createFeatures(video_filename, activity_name, activity_ind, feedBack, times, openface, lines, picture_sides):
    states = ["temp"]*len(times)
    o1,o2,o3,o4,o5 = [item[1] for item in openface], [item[2] for item in openface], [item[3] for item in openface], [item[4] for item in openface], [item[5] for item in openface]
    rows = zip(times,states, o1, o2, o3, o4, o5, lines, picture_sides)
    with open('tempData/features/'+ video_filename[-6:-4] + '_'+str(activity_ind)+'_' + activity_name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Video Filename", int(video_filename[-6:-4]), "Activity Int", activity_ind, "Activity Name", activity_name, "Feedback", feedBack, "Backbutton", "Unknown"])
        writer.writerow(["Time", "State", "Head Proximity", "Head Orientation", "Gaze Direction", "Eye Aspect Ratio", "Pupil Ratio", "Lines", "Picture Side"])
        for row in rows:
            writer.writerow(row)

def test():
    x = [1,2,3]
    y = [("bob","aoa"),("hi","bye"),("ding","honk")]
    l1, l2 = [item[0] for item in y], [item[1] for item in y]
    print(l1)
    print(l2)
    # rows = zip(x,z)
    # rows = zip(rows, x)
    # with open('tempData/test.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     for row in rows:
    #         writer.writerow(row)

def main():
    video_filenames = os.listdir('data/video')
    generateData("data/video/01.mp4")

main()
