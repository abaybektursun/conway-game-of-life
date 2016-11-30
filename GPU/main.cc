#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>
#include "routines.cc"

// parallel.cu  ----------------------------------------------------//
void game_of_life_cuda(const uchar1* const h_inWorld,               //
                       uchar1* const d_inWorld,                     //
                       uchar1* const d_outWorld,                    //
                       const size_t numRows, const size_t numCols); //
                                                                    //
void memoryManagement(const size_t numRows, const size_t numCols);  //
// parallel.cu  ----------------------------------------------------//

int main()
{
    uchar1 *h_inWorld,  *d_inWorld;
    uchar1 *h_outWorld, *d_outWorld;

    std::string in_file_name = "imgs/0.png";

    //load the image and get the pointers
    preProcess(&h_inWorld, &h_outWorld,
               &d_inWorld, &d_outWorld,
               in_file_name           );

    memoryManagement(numRows(), numCols());

    // CUDA
    game_of_life_cuda(h_inWorld, d_inWorld, d_outWorld, numRows(), numCols());
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    size_t numPixels = numRows()*numCols();
    checkCudaErrors(cudaMemcpy(h_outWorld, d_outWorld__, sizeof(uchar1) * numPixels, cudaMemcpyDeviceToHost));

    postProcess("imgs/2.png",h_outWorld);

    cleanUp();
    return 0;
}
