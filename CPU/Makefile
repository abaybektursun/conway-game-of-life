OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

#GCC_OPTS=-std=c++1y -O3 -Wall -Wextra -m64
GCC_OPTS=-std=c++1y -O1 -Wall -Wextra -m64


#student: main.o student_func.o compare.o reference_calc.o Makefile
#	$(NVCC) -o HW2 main.o student_func.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)

serial.o: serial.cc
	g++ -o serial.o serial.cc $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)
    

#clean:
#	rm -f *.o *.png hw
