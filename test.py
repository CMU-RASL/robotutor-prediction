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

print(compareFiles("data/video/18.mp4"))
