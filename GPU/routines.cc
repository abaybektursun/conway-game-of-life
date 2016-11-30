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

uchar1 *d_inWorld__;
uchar1 *d_outWorld__;

size_t numRows() { return uImg8UC1.rows; }
size_t numCols() { return uImg8UC1.cols; }

void preProcess(uchar1 **h_inWorld, uchar1 **h_outWorld,
                uchar1 **d_inWorld, uchar1 **d_outWorld,
                const std::string &filename)
{

    //Check the context initializes correctly
    checkCudaErrors(cudaFree(0));

    //----------------------------------------------------------------------------------------------
    // Read user's image
    cv::Mat userImg = cv::imread(filename,0);
    if (userImg.empty()) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }
    uImg8UC1 = cv::Mat(cv::Size(userImg.rows, userImg.cols),CV_8UC1);
    // Convert the user image to black and white using threshold
    cv::threshold(userImg, uImg8UC1, 128, 255, CV_THRESH_BINARY );
    cv::imwrite( "imgs/1.png",uImg8UC1 );
    //----------------------------------------------------------------------------------------------

    a_frame = cv::Mat(cv::Size(uImg8UC1.cols, uImg8UC1.rows),CV_8UC1);

    if (!uImg8UC1.isContinuous() || !a_frame.isContinuous()) {
        std::cerr << "uImg8UC1 is no continuous, closing the program" << std::endl;
        exit(1);
    }

    *h_inWorld  = (uchar1 *)uImg8UC1.ptr<uchar1>(0);
    *h_outWorld = (uchar1 *) a_frame.ptr<uchar1>(0);

    const size_t numPixels = numRows() * numCols();

    //allocate memory on the device for both input and output
    checkCudaErrors(cudaMalloc(d_inWorld, sizeof(uchar1) * numPixels));
    checkCudaErrors(cudaMalloc(d_outWorld, sizeof(uchar1) * numPixels));
    checkCudaErrors(cudaMemset(*d_outWorld, 0, numPixels * sizeof(uchar1))); //make sure no memory is left laying around
    //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(*d_inWorld, *h_inWorld, sizeof(uchar1) * numPixels, cudaMemcpyHostToDevice));
    d_inWorld__  = *d_inWorld;
    d_outWorld__ = *d_outWorld;

    /*checkCudaErrors(cudaMalloc(d_redBlurred,       sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_greenBlurred,     sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMalloc(d_blueBlurred,      sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));*/
}

void postProcess(const std::string& output_file, uchar1* data_ptr) {
    // TODO! To be implemented
    //cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
    //cv::Mat imageOutput;
    //cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
    //output the image
    //cv::imwrite(output_file.c_str(), imageOutput);
}

void cleanUp(void)
{
    cudaFree(d_inWorld__);
    cudaFree(d_outWorld__);
}
