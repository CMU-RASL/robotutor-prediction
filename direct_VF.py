import cv2

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

def main():
    print ("start")
