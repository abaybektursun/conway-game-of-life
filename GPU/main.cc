#include <iostream>
#include "utils.h"
#include <string>
#include <stdio.h>
#include <pthread.h>
#include <thread>
#include <string>
#include "routines.cc"

<<<<<<< HEAD
#define ITERS 1000
=======
#define ITERS 100
>>>>>>> 4d0f5286c686fe5775d7b33f8b65cd39b8edb158

// parallel.cu  ----------------------------------------------------//
void game_of_life_cuda(unsigned char* const d_inWorld,
                       unsigned char* const d_outWorld,
                       const size_t numRows,
                       const size_t numCols);
// parallel.cu  ----------------------------------------------------//

int main()
{
    unsigned char *h_inWorld,  *d_inWorld;
    unsigned char *h_outWorld, *d_outWorld;
    unsigned char *d_outWorld_swap;

    std::string in_file_name = "imgs/0.png";

    //load the image and get the pointers
    preProcess(&h_inWorld, &h_outWorld,
               &d_inWorld, &d_outWorld,
                      &d_outWorld_swap,
                          in_file_name);

    memoryManagement(&h_inWorld, &d_inWorld, &d_outWorld, &d_outWorld_swap);


    size_t numPixels = numRows()*numCols();
    // CUDA
    game_of_life_cuda(d_inWorld, d_outWorld, numRows(), numCols());
    // Get yo ass back to host
    checkCudaErrors(cudaMemcpy(h_outWorld, d_outWorld__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    unsigned i;
    for(i = 2; i < ITERS; i++){
        std::thread writer([&](){
            save8UC1Image(&h_outWorld, std::to_string(i) + ".png");
        });
        std::swap(d_inWorld, d_outWorld);

        game_of_life_cuda(d_inWorld, d_outWorld, numRows(), numCols());
        // Get yo ass back to host
        checkCudaErrors(cudaMemcpy(h_outWorld, d_outWorld__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

        writer.join();
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    save8UC1Image(&h_outWorld, std::to_string(i) + ".png");


    /*
    size_t numPixels = numRows()*numCols();
    checkCudaErrors(cudaMemcpy(h_outWorld, d_outWorld__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));
    postProcess("imgs/2.png",h_outWorld);
    checkCudaErrors(cudaFree(d_inWorld));
    checkCudaErrors(cudaFree(d_outWorld));
    */

    free(h_outWorld);
    cleanUp();
    return 0;
}
