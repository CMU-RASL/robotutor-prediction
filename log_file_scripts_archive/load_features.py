import os
from Context import Context
from VidFeature import VidFeature
from ActivityFeatures import ActivityFeatures
import csv
import cv2
import numpy as np
from datetime import datetime, timedelta

def get_matched_info(json_filenames, video_filenames):

    matched_info = []
    for json_name in json_filenames:
        res = list(filter(lambda x: json_name[:-5] in x, video_filenames))
        if len(res) == 1:
            matched_info.append((json_name, res[0]))

    return matched_info

def get_video_timestamps(matched_info):

    timestamps = []
    inds = []
    with open('data/video_timestamps.txt', mode='r') as f:
        for ii, line in enumerate(f):
            sep = line.index(',')
            inds.append(line[:sep])
            timestamps.append(line[sep+1:])

    final_timestamps = []
    for filenames in matched_info:
        cur_ind = inds.index(filenames[1][:-4])
        if cur_ind > -1:
            final_timestamps.append(timestamps[cur_ind])

    return final_timestamps


def process_json(json_filename):

    #Create JSON object
    cur_Context = Context(json_filename)

    with open('data/json/'+json_filename, 'r') as f:
        for ii, line in enumerate(f):
            if ii > 0 and len(line) > 40:
                cur_Context.add_line(line)
            if ii % 1000 == 0:
                print('Reading json', ii)

    cur_Context.print_context()

    return cur_Context

def process_video(video_filename, timestamp, side_arr):
    csv_filename = video_filename[:-4] + '_crop.csv'

    cur_VidFeature = VidFeature(video_filename, timestamp)

    with open('data/openface/' + csv_filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for ii, line in enumerate(csv_reader):
            if ii == 0:
                headers = [s.strip() for s in line]
            else:
                cur_VidFeature.add_line(headers, line)

            if ii % 1000 == 0:
                print('Reading openface', ii)

    cur_VidFeature.add_sides(side_arr)

    cur_VidFeature.print_features()

    return cur_VidFeature

def video_activity_begin(video_filename):

    cap = cv2.VideoCapture('data/video/'+video_filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    calc_timestamps = [0.0]
    frames = []
    pixel_counts = []

    gray_pix_val = 136

    for ii in range(400):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        center_wind = gray[gray.shape[0]//2 - 100: gray.shape[0]//2 + 100,
                           gray.shape[1]//2 - 100: gray.shape[1]//2 + 100]


        cur_count = np.count_nonzero(center_wind.flatten() == gray_pix_val)

        frames.append(gray)
        calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
        pixel_counts.append(cur_count)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    frames = np.array(frames)
    calc_timestamps = np.array(calc_timestamps)
    pixel_counts = np.array(pixel_counts)

    inds = np.where(pixel_counts > 1000)[0]

    new_timestamp = calc_timestamps[inds[-1]]

    return new_timestamp

def picture_side(video_filename):

    cap = cv2.VideoCapture('data/video/'+video_filename)

    sides = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        height, width, channels = frame.shape

        left_wind = frame[height//2 - 200 : height//2 + 200,
                          width // 4 - 200 : width // 4 + 200,:]

        right_wind = frame[height//2 - 200 : height//2 + 200,
                           3 * width // 4 - 200 : 3 * width // 4 + 200,:]

        colors, count = np.unique(left_wind.reshape(-1,left_wind.shape[-1]),
                                  axis=0, return_counts=True)
        left = colors[count.argmax()]

        colors, count = np.unique(right_wind.reshape(-1,right_wind.shape[-1]),
                                  axis=0, return_counts=True)
        right = colors[count.argmax()]

        if np.sum(left) > np.sum(right):
            sides.append('right')
        else:
            sides.append('left')

        if len(sides) % 100 == 0:
            print(len(sides))


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    f = open('data/picture_sides/' +
             video_filename[:2]+'_picture_sides.txt', 'w+')

    f.write('Filename: ' + video_filename + '\n')
    for side in sides:
        f.write(side + ' \n')
    f.close()
    print('Wrote to data/picture_sides/' + video_filename[:2]+
          '_picture_sides.txt')

def load_picture_sides(video_filename):
    sides = []
    with open('data/picture_sides/' + video_filename[:2]+
              '_picture_sides.txt', 'r') as f:
        for ii, line in enumerate(f):
            if ii > 0:
                sides.append(line.rstrip())

    return sides


def get_activityfeature(activity, events, vid_feat, activity_ind):

    if 'feedback' in activity.keys() and 'stop' in activity.keys():
        cur_vidfeatures = vid_feat.get_activity_features(activity['start'],
                                                         activity['stop'])

        if len(events) > 2:
            cur_activityfeatures = ActivityFeatures(activity,
                                                    vid_feat.filename)
            cur_activityfeatures.add_features(events, cur_vidfeatures)
            cur_activityfeatures.print_features(activity_ind)
    else:
        return None

def main():
    #Get list of filenames
    json_filenames = os.listdir('data/json')
    video_filenames = os.listdir('data/video')

    #Match jsons and videos
    matched_info = get_matched_info(json_filenames, video_filenames)

    print('Matched jsons and videos')

    #Get video timestamps
    video_timestamps = get_video_timestamps(matched_info)

    print('Got timestamps for videos')

    for jj in [1]: #range(len(matched_info)):
        print('--------------------------')
        print('Current Video is', matched_info[jj][1])

        if os.path.isfile('data/picture_sides/' + matched_info[jj][1][:2]+
                          '_picture_sides.txt'):
            sides = load_picture_sides(matched_info[jj][1])
        else:
            print('Picture Sides does not exist, creating file now')
            picture_side(matched_info[jj][1])


        print('Loaded Picture Side Info')

        activity_start_ms = video_activity_begin(matched_info[jj][1])
        activity_start_video = datetime.strptime(video_timestamps[jj].rstrip(),
                                                 "%Y-%m-%d-%H-%M-%S") + \
                                                 timedelta(milliseconds=
                                                           activity_start_ms)

        print('Found start of first activity in video')

        cur_Context = process_json(matched_info[jj][0])

        print('Created Context object')

        activity_start_log = cur_Context.get_activity_timestamp(0)

        diff = (activity_start_log - activity_start_video).total_seconds()
        new_video_start = datetime.strptime(video_timestamps[jj].rstrip(), \
                                            "%Y-%m-%d-%H-%M-%S") + \
                                            timedelta(seconds=diff)

        print('Calculated actual video start time')

        cur_VidFeature = process_video(matched_info[jj][1], new_video_start,
                                       sides)
        print('Created Video Feature Object')

        for ii in range(len(cur_Context.activities)):
            print('Current Activity is', cur_Context.activities[ii]['name'])

            get_activityfeature(cur_Context.activities[ii],
                                cur_Context.events[ii], cur_VidFeature, ii)
    print('Done')

if __name__ == "__main__":
    main()
