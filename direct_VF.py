import os
import cv2
import csv
import numpy as np
import VF_helpers as hp
import sys

def get_picture_side(video_filename): #create picture side text files
    print(video_filename)
    cap = cv2.VideoCapture('data/video/'+video_filename) #video name
    count = 1
    f = open('data/picture_sides/' + video_filename[:-4] +'_picture_sides.txt', 'w+')
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        f.write(str(hp.get_picture_side(frame))+"\n")
        count += 1
        if count >= 1000:
          count = 1
          f.flush()
    cap.release()
    cv2.destroyAllWindows()
    f.close()

def check_path(video_filename): #makes sure necessary files exist
    picture = os.path.exists('data/picture_sides/'+video_filename+'_picture_sides.txt')
    openface = os.path.exists('data/openface/' + video_filename+'.csv')
    if picture and openface:
        return True
    return False

def generateData(dir, video_num): #generate csv files

    picture_sides_path = '{}/picture_sides/{}_picture_sides.txt'.format(dir, video_num)
    if not os.path.exists(picture_sides_path):
        print('/t Could not find Picture sides')
        return
    with open(picture_sides_path) as f:
        all_sides = f.readlines()

    for i in range(1, len(all_sides)):
        if all_sides[i].strip() == "0" or all_sides[i].strip() == "left":
            all_sides[i] = 0
        elif all_sides[i].strip() == "1" or all_sides[i].strip() == "right":
            all_sides[i] = 1
        else:
            print('\tUnexpected character in picture sides', all_sides[i])
            return
        if i%5000 == 0:
            print('\tFinished Picture Sides', i)

    headers = ""
    all_openface = []
    openface_path = '{}/openface/{}_crop.csv'.format(dir, video_num)
    if not os.path.exists(openface_path):
        openface_path = '{}/openface/{}.csv'.format(dir, video_num)
        if not os.path.exists(openface_path):
            print('/t Could not find Openface')
            return
    with open(openface_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        nul = 0
        try: #tries to read csv files, aborts if encounters null character
            for ii, line in enumerate(csv_reader):
                nul = ii+1
                if ii == 0:
                    headers = [s.strip() for s in line]
                else:
                    all_openface.append(hp.get_openface_features(headers, line))
                if ii%1000 == 0:
                    print('\tFinished Open Face', ii)
        except:
            print("\tnull character found on line " + str(nul))

    #initial values
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
    press = False #is backbutton being pressed

    vid_path = '{}/videos/{}.mp4'.format(dir, video_num)
    if not os.path.exists(vid_path):
        return
    cap = cv2.VideoCapture(vid_path) #video name
    fps = cap.get(cv2.CAP_PROP_FPS)

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
            print("")
            print("\tactivity started", activity_name)
            #reset all variables
            frame_count = 0
            times = []
            vid_times = []
            picture_sides = []
            carry = 0
            lines = []
            openface = []
            activity_start = False
            backbutton = 0
            press = False
            circle = None
            frac_change = False

        if activity_name == "":
            continue

        if record:  #useful frames
            frame_count += 1
            times.append(frame_count/fps)
            vid_times.append(totalframe_count/fps)
            if totalframe_count-1 < len(all_openface):
                openface.append(all_openface[totalframe_count-1])
            else:
                print("\tnot enough open face features, aborting rest of video")
                break
            if totalframe_count-1 < len(all_sides):
                picture_sides.append(all_sides[totalframe_count-1])
            else:
                print("\tpicture side file too small")
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

        if hp.backbutton_pressed(frame):
            if not press:
                press = True
                backbutton = totalframe_count
                circle = frame
            else:
                backbutton = totalframe_count
                circle = frame
        else:
            press = False

        fb = hp.feedBackType(frame)
        if fb != None and not seenfeedBack and record:
            if backbutton != 0 and totalframe_count-backbutton <= 300:
                backbutton = 1
            else:
                backbutton = 0
            feedBack_num += 1
            seenfeedBack = True
            #activity ended, create csv file
            createFeatures(dir, video_num, activity_name, feedBack_num, fb, times, vid_times, openface, lines, picture_sides, backbutton)
            record = False
        elif fb == None and seenfeedBack:
            seenfeedBack = False
            activity_name = ""

        if totalframe_count%500 == 0:
            print('\tFinished Frame:', totalframe_count, 'Time (min):', np.round(totalframe_count/fps/60, decimals=2))

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def createFeatures(dir, video_num, activity_name, activity_ind, feedBack, times, vid_times, openface, lines, picture_sides, backbutton): #creates csv files
    if times[-1] < 50:
        backbutton = 0
    if activity_name == 'story_hear':
        activity_type = 0
    elif activity_name == 'story_read':
        activity_type = 1
    else:
        activity_type = 2

    output_path = '{}/csvs/{}_{}_{}.csv'.format(dir, video_num, activity_ind, activity_name)
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Video Filename", video_num, "Activity Ind", activity_ind, \
                "Activity Name", activity_name, "Feedback", feedBack, "Backbutton", backbutton])
        writer.writerow(["Time", "Activity Ind", "Video Time", "Head Proximity", \
                "Head Orientation", "Gaze Direction", "Eye Aspect Ratio", "Pupil Ratio", \
                "Confidence", "Success", "AU04", "AU07", "AU12", "AU25", "AU26", "AU45", \
                "Progress", "Picture Side", "Activity Type"])
        for i in range(len (times)):
            if openface[i][2] > np.pi:
                openface[i][2] -= 2*np.pi
            writer.writerow([times[i], activity_ind, vid_times[i], openface[i][1], \
                    openface[i][2], openface[i][3], openface[i][4], openface[i][5], openface[i][6], \
                    openface[i][7], openface[i][8], openface[i][9], openface[i][10], openface[i][11], \
                    openface[i][12],openface[i][13],lines[i], picture_sides[i], activity_type])

def main(dir, video_filename): #computes on specific file or directory
    print("starting video " + video_filename)
    generateData(dir, video_filename)
    # get_picture_side(dir+"/"+video_filename)

for n in range(99, 100):
    vid_name = '{0:04d}'.format(n)
    main('dataset2', vid_name)
