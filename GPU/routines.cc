#include <string>
#include "utils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

cv::Mat uImg_8U;

unsigned char *d_inWorld__;
unsigned char *d_outWorld__;

size_t numRows() { return uImg_8U.rows; }
size_t numCols() { return uImg_8U.cols; }


unsigned char *MatToUnsignChar(cv::Mat inMat){
    int height = numRows();
    int width  = numCols();
    //unsigned char buffer[height * width];
    unsigned char *buffer = (unsigned char *)malloc(sizeof(unsigned char) *  numRows() * numCols());
    uchar* p;
    for (int i = 0; i < height; ++i) {
        p = inMat.ptr<uchar>(i);
        for (int j = 0; j < width; ++j) {
            buffer[i * width + j] = p[j];
        }
    }
    return buffer;
}

void preProcess(unsigned char **h_inWorld, unsigned char **h_outWorld,
                const std::string &filename)
{
    //Check the context initializes correctly
    checkCudaErrors(cudaFree(0));

    //----------------------------------------------------------------------------------------------
    // Read user's image
    cv::Mat userImg = cv::imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
    if (userImg.empty()) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }
    //uImg_8U = cv::Mat(cv::Size(userImg.cols, userImg.rows),CV_8UC1);
    // Convert the user image to black and white using threshold
    //cv::threshold(userImg, uImg_8U, 128, 255, CV_THRESH_BINARY );
    uImg_8U = userImg > 128;
    cv::imwrite( "imgs/1.png",uImg_8U );
    //----------------------------------------------------------------------------------------------

    if (!uImg_8U.isContinuous()) {
        std::cerr << "uImg_8U is no continuous, closing the program" << std::endl;
        exit(1);
    }

     *h_inWorld  = MatToUnsignChar(uImg_8U);

     *h_outWorld =  (unsigned char *)malloc(sizeof(unsigned char) *  numRows() * numCols());
}

void memoryManagement(unsigned char **h_inWorld,  unsigned char **d_inWorld,
                                                  unsigned char **d_outWorld)
{
    const size_t numPixels = numRows() * numCols();

    //allocate memory on the device for both input and output
    checkCudaErrors(cudaMalloc(d_inWorld,      sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_outWorld,     sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_outWorld, 0, sizeof(unsigned char) * numPixels));
    //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(*d_inWorld, *h_inWorld, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
    d_inWorld__  = *d_inWorld;
    d_outWorld__ = *d_outWorld;

}

void save8UImage(unsigned char **ucharArr, std::string filename)
{
    //cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)*ucharArr);

    //cv::Mat imageOutput;
    //cv::cvtColor(output, imageOutput, CV_RGBA2BGR);
    //output the image
    //cv::imwrite("imgs/" + filename, imageOutput);
    //---------------------------------------------------------------------
    cv::Mat outMat = cv::Mat(cv::Size(uImg_8U.cols, uImg_8U.rows),CV_8U, *ucharArr);
    cv::imwrite( "imgs/" + filename, outMat );
}

void cleanUp()
{
    cudaFree(d_inWorld__);
    cudaFree(d_outWorld__);
}
