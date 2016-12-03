# Set these parameters below (Links to input images) ------------------------------------------------
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IMG_LARGE=http://img15.deviantart.net/5491/i/2015/172/b/9/norwegian_wood_by_land_walker-d8y5dn4.png
IMG_SMALL=http://www.verdenslitteratur.dk/files/8014/0053/3569/Haruki_Murakami_468432a_sh_381.png
IMG_LARGE_SPACE=http://i.imgur.com/7PfNzTO.png
# Pick either small or large image ------------------------------------------------------------------
IMG=$(IMG_LARGE)
IMGS_FOLDER=imgs
# Set the command for nvcc compiler -----------------------------------------------------------------
NVCC=nvcc
CUDA_INCLUDEPATH=${CUDA_HOME}/include
#----------------------------------------------------------------------------------------------------
OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui
#----------------------------------------------------------------------------------------------------
#GCC_OPTS=-std=c++1y -O1 -Wall -Wextra -m64
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64 -std=c++1y
#----------------------------------------------------------------------------------------------------
APPLICATION_NAME=game_of_life_cuda
SPEED_VIDEO=1/0.08

all: imgs_folder get_image master create_run

#directories: $(IMGS_FOLDER)

imgs_folder:
	mkdir -p $(IMGS_FOLDER)

get_image:
	-wget -O $(IMGS_FOLDER)/0.png $(IMG)

master: main.o device.o Makefile
	$(NVCC) -o $(APPLICATION_NAME) main.o device.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cc utils.h routines.cc
	g++ -c main.cc $(GCC_OPTS) -I $(OPENCV_INCLUDEPATH) -I $(CUDA_INCLUDEPATH)

device.o: device.cu utils.h
	nvcc -c device.cu $(NVCC_OPTS)

create_run:
	@echo "./game_of_life_cuda && \
<<<<<<< HEAD
	ffmpeg -r 24 -f image2 -r $(SPEED_VIDEO) -i $(IMGS_FOLDER)/%d.png -vcodec mpeg4 -y $(APPLICATION_NAME).mp4 && \
=======
	ffmpeg -r 24 -f image2 -r 1/0.4 -i $(IMGS_FOLDER)/%d.png -vcodec mpeg4 -y $(APPLICATION_NAME).mp4 && \
>>>>>>> 4d0f5286c686fe5775d7b33f8b65cd39b8edb158
	echo \"Success! Video '$(APPLICATION_NAME).mp4' was generated\"" > run.sh \
	&& chmod +x run.sh

clean:
	rm -f run.sh *.o *.log $(IMGS_FOLDER)/*.png $(APPLICATION_NAME)
