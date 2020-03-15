import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import helpers as hp
import read_data as rd

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

    f = open('data/picture_sides/'+video_filename[-6:-4]+'_picture_sides.txt')
    all_sides = f.readlines()
    f.close()
    all_sides = all_sides[1:]
    for i in range(len(all_sides)):
        if all_sides[i].strip() == "left":
            all_sides[i] = 0
        elif all_sides[i].strip() == "right":
            all_sides[i] = 1
        else:
            print(all_sides[i])
            error()

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
    activity_start = False #true means activity has started ans false means ended
    frame_count = 0
    totalframe_count = 0 #use this to extract precomputed data
    activity_name = ""
    times = []
    openface = []
    picture_sides = []
    lines = []
    carry = 0
    frac_change = False
    record = False #skips useless frames

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        totalframe_count += 1

        if not record and not seenfeedBack: #finished with feedback screen
            activity = hp.get_activity_type(frame)
            if activity != 'n/a':
                activity_name = activity
            elif activity_name != "": #activity name found and activity starting
                activity_start = True
                record = True

        if activity_start: #activity starting
            print("activity started")
            #reset all variables
            frame_count = 0
            times = []
            picture_sides = []
            carry = 0
            lines = []
            openface = []
            activity_start = False

        if record:  #useful frames
            frame_count += 1
            times.append(frame_count/fps)
            if totalframe_count-1 < len(all_openface):
                openface.append(all_openface[totalframe_count-1])
            else:
                print("not enough open face features, aborting rest of video")
                break
            if totalframe_count-1 < len(all_sides):
                picture_sides.append(all_sides[totalframe_count-1])
            else:
                print("picture side file too small")
                picture_sides.append(hp.get_picture_side(frame))
            frac = round(hp.get_read_fraction(frame, picture_sides[len(picture_sides)-1]), 2)
            if frac == 1:
                frac_change = True
            if frac != 1 and frac_change:
                frac_change = False
                carry += 1
            nextFrac = frac+carry
            if len(lines) == 0 or nextFrac >= lines[len(lines)-1]:
                lines.append(nextFrac)
            else:
                lines.append(lines[len(lines)-1])


        # if activity_start == False: #activity ended
        #     createFeatures(video_filename, activityname_name, feedBack_num, allfeedBack[len(allfeedBack)-1], times)

        # height, width, channels = frame.shape
        # yellow = frame[height // 2 - 10 : height // 2 + 10, width // 4 - 25 : width // 4 + 25]
        # green = frame[height // 5 - 10 : height // 5 + 10, width // 4 - 25 : width // 4 + 25]
        # red = frame[4*height // 5 - 10 : 4*height // 5 + 10, width // 4 - 25 : width // 4 + 25]

        # snap = frame[height//2 - 50 : height//2 + 50,
        #                   width // 2 - 100 : width // 2 + 100,:]

        fb = feedBackType(frame)
        if fb != None and not seenfeedBack and record:
            print("feedback", fb)
            feedBack_num += 1
            seenfeedBack = True
            createFeatures(video_filename, activity_name, feedBack_num, fb, times, openface, lines, picture_sides)
            activity_name = ""
            record = False
        elif fb == None and seenfeedBack:
            seenfeedBack = False
    # When everything done, release the capture

    cap.release()
    cv2.destroyAllWindows()
    print(totalframe_count)

def createFeatures(video_filename, activity_name, activity_ind, feedBack, times, openface, lines, picture_sides):
    states = ["temp"]*len(times)
    o1,o2,o3,o4,o5 = [item[1] for item in openface], [item[2] for item in openface], [item[3] for item in openface], [item[4] for item in openface], [item[5] for item in openface]
    rows = zip(times,states, o1, o2, o3, o4, o5, lines, picture_sides)
    with open('tempData/compare/'+ video_filename[-6:-4] + '_'+str(activity_ind)+'_' + activity_name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Video Filename", int(video_filename[-6:-4]), "Activity Int", activity_ind, "Activity Name", activity_name, "Feedback", feedBack, "Backbutton", "Unknown"])
        writer.writerow(["Time", "State", "Head Proximity", "Head Orientation", "Gaze Direction", "Eye Aspect Ratio", "Pupil Ratio", "Lines", "Picture Side"])
        for row in rows:
            writer.writerow(row)

def main():
    video_filenames = os.listdir('data/video')
    for vf in video_filenames:
        if vf == "01.mp4":
            print("starting video " + vf)
            generateData("data/video/"+vf)

main()
