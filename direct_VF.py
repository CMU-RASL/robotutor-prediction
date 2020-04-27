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

def generateData(video_filename): #generate csv files
    vf = video_filename.split("/")[-1][:-4]
    if not check_path(vf):
        print("missing picture or openface file")
        return

    cap = cv2.VideoCapture(video_filename) #video name

    f = open('data/picture_sides/'+vf+'_picture_sides.txt')
    print(vf)
    all_sides = f.readlines() #starts reading from the first line
    f.close()
    for i in range(len(all_sides)):
        if all_sides[i].strip() == "0":
            all_sides[i] = 0
        elif all_sides[i].strip() == "1":
            all_sides[i] = 1
        else:
            print(all_sides[i])
            error()

    headers = ""
    all_openface = []
    with open('data/openface/' + vf+'.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        nul = 0
        try: #tries to read csv files, aborts if encounters null character
            for ii, line in enumerate(csv_reader):
                nul = ii+1
                if ii == 0:
                    headers = [s.strip() for s in line]
                else:
                    all_openface.append(hp.get_openface_features(headers, line))
        except:
            print("null character found on line " + str(nul))



    fps = cap.get(cv2.CAP_PROP_FPS)

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
            print("activity started", activity_name)
            #reset all variables
            frame_count = 0
            times = []
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
            createFeatures(vf, activity_name, feedBack_num, fb, times, openface, lines, picture_sides, backbutton)
            record = False
        elif fb == None and seenfeedBack:
            seenfeedBack = False
            activity_name = ""

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def createFeatures(vf, activity_name, activity_ind, feedBack, times, openface, lines, picture_sides, backbutton): #creates csv files
    states = ["temp"]*len(times)
    with open('data/activities/'+ vf + '_'+str(activity_ind)+'_' + activity_name + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Video Filename", int(vf), "Activity Int", activity_ind, "Activity Name", activity_name, "Feedback", feedBack, "Backbutton", backbutton])
        writer.writerow(["Time", "State", "Head Proximity", "Head Orientation", "Gaze Direction", "Eye Aspect Ratio", "Pupil Ratio", "Confidence", "Success", "AU04", "AU07", "AU12", "AU25", "AU26", "AU45", "Lines", "Picture Side"])
        for i in range(len (times)):
            writer.writerow([times[i], states[i], openface[i][1], openface[i][2], openface[i][3], openface[i][4], openface[i][5], openface[i][6], openface[i][7], openface[i][8], openface[i][9], openface[i][10], openface[i][11],openface[i][12],openface[i][13],lines[i], picture_sides[i]])

def main(dir, video_filename): #computes on specific file or directory
    if video_filename == "all":
        video_filenames = os.listdir(dir)
        for vf in video_filenames:
            print("starting video " + vf)
            generateData(dir+"/"+vf)
            # get_picture_side(dir+"/"+vf)
    else:
        print("starting video " + video_filename)
        generateData(dir+"/"+video_filename)
        # get_picture_side(dir+"/"+video_filename)

main(sys.argv[1], sys.argv[2])
