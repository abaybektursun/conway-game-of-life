OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

GCC_OPTS=-std=c++1y -O1 -Wall -Wextra -m64


serial: serial.cc
	g++ -o serial serial.cc $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)
    

#clean:
#	rm -f *.o *.png hw
