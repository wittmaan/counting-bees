# Counting Bees

## setup

todo

The video files are not stored at github.

## Extract frames

Run the script `01_extract_frames.sh` to extract frames from given video files.


## Negative Samples

To account for negative samples, the following steps are needed:

1. Draw a random bounding box with labelimg at the dedicated file.
2. Adjust the annotation xml-file to xmin=0, ymin=0, xmax=0 and ymax=0.


## TODO

- do augmentations on-the-fly
- put samples of extracted frames to github

