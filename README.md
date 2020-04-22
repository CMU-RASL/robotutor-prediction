# robotutor-analysis

## Setting up Remote Machine
1. Install [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) or other SSH client
2. Install [WinSCP](https://winscp.net/eng/index.php) or other file transfer application
3. Login to `bardot.autonomy.ri.cmu.edu`
4. Follow this [tutorial](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation) to install OpenFace into your home directory (i.e. create `~\OpenFace`)
5. Apply `chmod 777` to any directories that you will be adding/moving/deleting files from
6. Install ffmpeg

## Generating OpenFace files
1. Create directory `~\OpenFace\build\videos` and copy all videos into that directory
2. Create directory `~\OpenFace\build\videos\crop_videos`
3. In `~\OpenFace\build\videos`, execute the following:

  ```for f in *.mp4; do ffmpeg -i $f -filter:v "crop=185:135:839:20" crop_videos/$f; done;```
4. In `~\OpenFace\build\crop_videos`, execute the following:

  ```for f in *.mp4; do ../../bin/FeatureExtraction -f $f; done;```

5. Copy the csv files created in `~\OpenFace\build\crop_videos\processed` to your computer
