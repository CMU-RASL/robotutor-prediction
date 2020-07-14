# robotutor-prediction

Methodology described in R. Kaushik and R. Simmons, “Early Prediction of Student Engagement from Facial and Contextual Features,” in International Conference on Social Robotics, 2020.

### Setup
#### Setting up Remote Machine (Optional)
1. Install [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) or other SSH client
2. Install [WinSCP](https://winscp.net/eng/index.php) or other file transfer application
3. Login to `bardot.autonomy.ri.cmu.edu`
4. Follow this [tutorial](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation) to install OpenFace into your home directory (i.e. create `~\OpenFace`)
5. Apply `chmod 777` to any directories that you will be adding/moving/deleting files from

#### Required Installation
Install ffmpeg, python3, pickle, numpy, cv2, multiprocessing

### Dataprocessing

#### Generating OpenFace files
1. Create directory `~\OpenFace\build\videos` and copy all videos into that directory
2. Create directory `~\OpenFace\build\videos\crop_videos`
3. In `~\OpenFace\build\videos`, execute the following:

  ```for f in *.mp4; do ffmpeg -i $f -filter:v "crop=185:135:839:20" crop_videos/$f; done;```
4. In `~\OpenFace\build\crop_videos`, execute the following:

  ```for f in *.mp4; do ../../bin/FeatureExtraction -f $f; done;```

5. Copy the csv files created in `~\OpenFace\build\crop_videos\processed` to your computer

#### Generating Data From Videos
To generate data from videos, use the `direct_VF.py` script located in the home directory.

#### Getting Picture Sides
1. Create directory `~\data\picture_sides`
2. Use the function
```python
def get_picture_side(video_filename) :
```
to generate a file containing the side the picture is located on for each frame (represented by a line). This file will be stored in the data/picture_sides directory.

#### Generating CSV Files
Make sure video mp4, picture side, and OpenFace feature files are available.

1. Create directory `~\data\activities`
2. Run
```bash
python3 direct_VF directory_name video_name
```
3. If you want to generate CSV files for all videos in a given directory, run the following:
```bash
python3 direct_VF directory_name all
```
This will create csv files for each activity labeled by the video name, activity number, and activity name located in the data/activities directory.

#### Creating Dataset
Run `create_dataset.py` with appropriate variables to create a `pkl` file for the entire dataset


### Generating Models and Predicting Events
`main.py` is the file to run which:
1. Splits into train-test with cross-validation
2. Runs all combinations of hyperparameters and calculates performance metrics
3. Chooses hyperparameter combination with highest combined performance metric
4. Saves results into a `pkl` file
