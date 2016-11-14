# Set these parameters below (Links to input images) ------------------------------------------------
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IMG_LARGE=http://img15.deviantart.net/5491/i/2015/172/b/9/norwegian_wood_by_land_walker-d8y5dn4.png
IMG_SMALL=http://www.verdenslitteratur.dk/files/8014/0053/3569/Haruki_Murakami_468432a_sh_381.png
# Pick either small or large image ------------------------------------------------------------------
IMG=$(IMG_SMALL)
IMGS_FOLDER=imgs
# Set the command for nvcc compiler -----------------------------------------------------------------
NVCC=nvcc
CUDA_INCLUDEPATH=${CUDA_HOME}/include
#----------------------------------------------------------------------------------------------------
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
#----------------------------------------------------------------------------------------------------
GCC_OPTS=-std=c++1y -O1 -Wall -Wextra -m64
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
#----------------------------------------------------------------------------------------------------

all: imgs_folder get_image master run

#directories: $(IMGS_FOLDER)

imgs_folder:
	mkdir -p $(IMGS_FOLDER) 

get_image:
	wget -O $(IMGS_FOLDER)/0.png $(IMG)

master: main.o parallel.o
	$(NVCC) -o game_of_life_cuda main.o parallel.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cc utils.h routines.cc
	g++ -c main.cc $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -I $(CUDA_INCLUDEPATH)

parallel.o: parallel.cu utils.h
	nvcc -c parallel.cu $(NVCC_OPTS)

run:
	mv .run.sh run.sh


clean:
	rm -f *.o *.png hw