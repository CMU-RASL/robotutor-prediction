import helpers as hp
import cv2


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
        hp.get_activity_type(frame)
        break
    cap.release()
    cv2.destroyAllWindows()


#print(compareFiles("data/video/18.mp4"))
saveFrame('tempData/storyread.mp4')
