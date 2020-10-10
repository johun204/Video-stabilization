# Video stabilization
Video stabilization using Harris Corner Detection.

## Overview
<img src='https://github.com/johun204/Video-stabilization/raw/main/media/image1.gif'>

 * The Stabilization Rectangle was shifted according to the position change detected using the Histogram of Oriented Gradient(HOG) to calculate the similarity between the previous frame and the current frame corner detected by Harris Corner Detection.

 * Improved Histogram of Oriented Gradient(HOG) calculation speed using Integral Image.

## Result
<img src='https://github.com/johun204/Video-stabilization/raw/main/media/image2.gif'>

## Requirements

* OpenCV >= 3.4
* C++
