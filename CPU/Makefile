OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

GCC_OPTS=-std=c++1y -O1 -Wall -Wextra -m64

IMGS_FOLDER=imgs

all: directories serial run video

directories: $(IMGS_FOLDER)

imgs:
	mkdir $(IMGS_FOLDER)

serial: serial.cc
	g++ -o serial serial.cc $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)

run:
	./serial

video:
	ffmpeg -r 24 -f image2 -r 1/0.08 -i $(IMGS_FOLDER)/%d.png -vcodec mpeg4 -y movie.mp4

#clean:
#	rm -f *.o *.png hw

#ffmpeg -f image2 -r 1/5 -i %d.png -vcodec mpeg4 -y movie.mp4
#ffmpeg -r 24 -f image2 -r 1/0.08 -i %d.png -vcodec mpeg4 -y movie.mp4

