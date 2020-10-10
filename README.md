# Video stabilization
Video stabilization using Harris Corner Detection.

## Overview
<img src='https://github.com/johun204/Video-stabilization/raw/main/media/image1.gif'>

 * After detecting a corner using Harris Corner Detection in the previous frame and the current frame, Move the Stabilization Rectangle by comparing the similarity using Histogram of Oriented Gradient (HOG).

 * Improved Histogram of Oriented Gradient(HOG) calculation speed using Integral Image.

## Result
<img src='https://github.com/johun204/Video-stabilization/raw/main/media/image2.gif'>

## Requirements

* OpenCV >= 3.4
* C++
