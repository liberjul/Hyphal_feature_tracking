# Hyphal_feature_tracking

This repository employs [TensorFlow's](https://github.com/tensorflow/tensorflow) object detection algorithm to identify and track growing tips of fungi or similar biological objects.

Requirements:
- Python >=3.6
- TensorFlow >=1.4
- Matplotlib
- Numpy
- Pillow 1.0
- Imageio
- Scipy
- Pandas
- Anaconda3 (preferred)


## Installation
```
git clone https://github.com/liberjul/Hyphal_feature_tracking
```

### If using conda (installed from Anaconda):
You can do this easiest in the Anaconda Prompt
```
conda create -n hf_tracking python=3.7
conda activate hf_tracking
conda install tensorflow
conda install matplotlib
conda install numpy
conda install pillow
conda install imageio
```

Download object_detection models
```
git clone https://github.com/tensorflow/models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
conda develop .
```

### If not using conda
```
pip install tensorflow
pip install matplotlib
pip install numpy
pip install pillow
pip install imageio

git clone https://github.com/tensorflow/models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
More detailed instructions can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

## The Data

The model was developed using ~ 1000 annotations of hyphal tips in micrographs. Therefore, you should be aiming to detect similar-looking objects in you images.

### What you need to know for accurate metrics:
 - What are the physical dimensions captured by each frame, in &mu;m
 - How many minutes are there between frames
 
Edit the `use_model.py` file to add your values from above under `FRAME_LENGTH`, `FRAME_WIDTH`, annd `FRAME_TIME`.
You can adjust the values of `CONF_THR` and `OUTLIER_PROP` as you see fit.
To change the folder destinations, edit any of the variables beginning with `PATH`.

### The Images
Make sure the image names are in this format: `<image_prefix>XXX.jpg`, where the `image_prefix` is constant and the `XXX` are chronological image numbers. Look at the example images included in test ims.

## Testing

From the main directory, you should be able to run:

```
python use_model.py JU15B_MEA_60000_4_
```
You output should resemble something like this:
```
2019-12-12 11:15:09.988065: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
./test_ims\JU15B_MEA_60000_4_078.jpg
./test_ims\JU15B_MEA_60000_4_079.jpg
./test_ims\JU15B_MEA_60000_4_080.jpg
./test_ims\JU15B_MEA_60000_4_081.jpg
./test_ims\JU15B_MEA_60000_4_082.jpg
./test_ims\JU15B_MEA_60000_4_083.jpg
./test_ims\JU15B_MEA_60000_4_084.jpg
./test_ims\JU15B_MEA_60000_4_085.jpg
./test_ims\JU15B_MEA_60000_4_086.jpg
./test_ims\JU15B_MEA_60000_4_087.jpg
```
You will see these new files in the directory:
```
box_data_JU15B_MEA_60000_4_.csv
JU15B_MEA_60000_4_speed_data.csv
JU15B_MEA_60000_4_speed_distribution.jpg
JU15B_MEA_60000_4_speed_vs_time.jpg
```

There will also be two `.mp4` files in the `model_annots/` directory and two subfolders in `model_annots/` with the images used to make the videos.

## Using on your own images

Add your image files to a directory specified by `PATH_TO_IMS` in `use_model.py`. The default is `test_ims/`.

Run the command based on your image file prefixes:
```
python use_model.py <image_prefix>
```

### What is the output data?

The file `<image_prefix>speed_data.csv` contains data on each measured distance increment. Time is in minutes, speed is in &mu;m/min, and the y and x components are normalized direction vectors.

### How do I adapt this?

I will be adding scripts for training on new images soon, but my method is based on this [tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).

The image dimensions, labels for plots, and statistics can all be changed. Please send any recommendations you may have.
