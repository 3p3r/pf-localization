# pf-localization
Localization using a Particle Filter (and random walk model)
![Localization demo](data/demo.gif?raw=true)

## What is this?
This is a Cuda-C++ library to localize a camera, pointed at a checkerboard pattern without solving the [PnP problem](https://en.wikipedia.org/wiki/Perspective-n-Point) and using a particle filter. Solving the PnP problem is not always feasible (specifically in case of real-time SLAM), therefore this project aims at localizing the camera rotation and translation with a [Random Walk model](https://en.wikipedia.org/wiki/Random_walk).

## Building
In order to build this library, you need a system with CMake and Cuda SDK 7+ installed.

```bash
cd <project root>
mkdir build;
cd build;
cmake .. -G "Visual Studio 14 Win64"
cmake --build . --config Release
cmake --build . --target INSTALL
```

## Running
Matlab is required (but not necessary) to run this library with a sample dataset (in `data/board.mp4`). The Matlab script that uses this library is `main.m`. The first time you run the Matlab file, it'll take some time to build a cache of video's ground truth data using PnP solver of Matlab.

Matlab code also ships with a Matlab implementation of the project. However, If you like to run this code with Cuda, you must have a Nvidia Cuda-capable graphics card.


## Reference
This code is an adaptation of the theories provided in the following paper:

```
@article{doi: 10.1117/1.JEI.23.1.013029,
author = { Seok-Han  Lee},
title = {Real-time camera tracking using a particle filter combined with unscented Kalman filters},
journal = {Journal of Electronic Imaging},
volume = {23},
number = {},
pages = {23 - 23 - 19},
year = {2014},
doi = {10.1117/1.JEI.23.1.013029},
URL = {http://dx.doi.org/10.1117/1.JEI.23.1.013029},
eprint = {}
}
```
