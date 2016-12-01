#include <string>
#include "utils.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

cv::Mat uImg8UC1;
cv::Mat a_frame;

unsigned char *d_inWorld__;
unsigned char *d_outWorld__;

size_t numRows() { return uImg8UC1.rows; }
size_t numCols() { return uImg8UC1.cols; }


unsigned char *MatToUnsignChar(cv::Mat inMat){
    int height = numRows();
    int width  = numCols();
    unsigned char buffer[height * width];
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
                unsigned char **d_inWorld, unsigned char **d_outWorld,
                unsigned char **d_outWorld_swap,
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
    //uImg8UC1 = cv::Mat(cv::Size(userImg.cols, userImg.rows),CV_8UC1);
    // Convert the user image to black and white using threshold
    //cv::threshold(userImg, uImg8UC1, 128, 255, CV_THRESH_BINARY );
    uImg8UC1 = userImg > 128;
    cv::imwrite( "imgs/1.png",uImg8UC1 );
    //----------------------------------------------------------------------------------------------

    // A single interations will be stored here
    a_frame = cv::Mat(cv::Size(uImg8UC1.cols, uImg8UC1.rows),CV_8U);

    if (!uImg8UC1.isContinuous() || !a_frame.isContinuous()) {
        std::cerr << "uImg8UC1 is no continuous, closing the program" << std::endl;
        exit(1);
    }

     *h_inWorld  = MatToUnsignChar(uImg8UC1);

     *h_outWorld      =  (unsigned char *)malloc(sizeof(unsigned char) *  numRows() * numCols());
     *d_outWorld_swap =  (unsigned char *)malloc(sizeof(unsigned char) *  numRows() * numCols());
}

void memoryManagement(unsigned char **h_inWorld,  unsigned char **d_inWorld,
                      unsigned char **d_outWorld, unsigned char **d_outWorld_swap)
{
    const size_t numPixels = numRows() * numCols();

    //allocate memory on the device for both input and output
    checkCudaErrors(cudaMalloc(d_inWorld,       sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_outWorld,      sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_outWorld_swap, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_outWorld,      0, numPixels * sizeof(unsigned char)));
    checkCudaErrors(cudaMemset(*d_outWorld_swap, 0, numPixels * sizeof(unsigned char)));
    //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(*d_inWorld, *h_inWorld, sizeof(unsigned char) * numPixels, cudaMemcpyHostToDevice));
    d_inWorld__  = *d_inWorld;
    d_outWorld__ = *d_outWorld;

}

void save8UC1Image(unsigned char **ucharArr, std::string filename)
{
    //cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)*ucharArr);

    //cv::Mat imageOutput;
    //cv::cvtColor(output, imageOutput, CV_RGBA2BGR);
    //output the image
    //cv::imwrite("imgs/" + filename, imageOutput);
    //---------------------------------------------------------------------
    cv::Mat outMat = cv::Mat(cv::Size(uImg8UC1.cols, uImg8UC1.rows),CV_8U, *ucharArr);
    cv::imwrite( "imgs/" + filename, outMat );
}

void cleanUp()
{
    cudaFree(d_inWorld__);
    cudaFree(d_outWorld__);
}
