# extract_thermal_image repo
This package is used for shimizu project of thermal image preprocessing, ir-rgb calibration and shimizu time-synced thermal extraction.


## 1. introduction
This repo needs OpenCV:

- 1.1. **IR_preprocessing** is used to convert 16 bit raw thermal image to 8 bit image;
- 1.2 **IR_RGB_calib** is used to extract pairwise rgb-thermal image for [only] extrinsic calibration;
- 1.3 **thermal_extraction_node** is used to extract the corresponding thermal images for shimizu RGB images;
- 1.4 **thermal_intensity_correction** is used to equalize the intensity of thermal images across all images.


## 2. How to use it?
Build:
```
cd ~/catkin_ws/src
git clone https://github.com/levenberg/extract_thermal_image.git
cd ..
catkin_make
```
Run:

First customize the data in the corresponding *.launch*, then
```
roslaunch extract_thermal_image *.launch
``` 
## 3. Contact
Huai Yu (huaiy@andrew.cmu.edu)

