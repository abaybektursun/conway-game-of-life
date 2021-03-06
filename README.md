# Conway's Game of Life


> "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970"

>-*Wikipedia*

This project has two different implementations of Conway's Game of Life, serial version for CPU and parallel **CUDA** version for (NVIDIA) GPUs. Both implementations take in an png image as an input, turn it into thresholded black and white image, and compute the game of life for desired amount of iterations producing mp4 video as an output.

Example screenshots:
![conway_gol](http://i.imgur.com/5NgOIli.png "Conway's Game of Life")

#### Instructions:
To compile and run, follow these commands
```bash
$ cd GPU # or CPU
$ make
$ ./run.sh
```
#### Requirements and Dependencies:
* Linux System
* NVIDIA GPU
* CUDA Toolkit
* OpenCV
* FFmpeg
* Standard developer build tools (like g++)
