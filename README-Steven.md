# robotutor-analysis

## Generating Data From Videos

To generate data from videos, use the direct_VF.py script located in the home directory.

### Getting Picture Sides

Create directory `~\data\picture_sides`
Then, use the function
```python
def get_picture_side(video_filename) :
```
to generate a file containing the side the picture is located on for each frame (represented by a line). This file will be stored in the data/picture_sides directory.

### Generating CSV Files

Make sure video mp4, picture side, and OpenFace feature files are available.

Create directory `~\data\activities`
Run
```bash
python3 direct_VF directory_name video_name
```
If you want to generate CSV files for all videos in a given directory, run the following:
```bash
python3 direct_VF directory_name all
```
This will create csv files for each activity labeled by the video name, activity number, and activity name located in the data/activities directory.
