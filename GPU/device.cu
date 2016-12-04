#include "utils.h"
#include <stdio.h>


#define live_cells( thread_1D_pos, x, y, world ) (    world[(thread_1D_pos-x-1)%(x*y)] + world[(thread_1D_pos-x)%(x*y)] + world[(thread_1D_pos-x+1)%(x*y)]     \
                                   +                  world[(thread_1D_pos - 1)%(x*y)] +                                  world[(thread_1D_pos-x+1)%(x*y)]     \
                                   +                  world[(thread_1D_pos+x-1)%(x*y)] + world[(thread_1D_pos+x)%(x*y)] + world[(thread_1D_pos+x+1)%(x*y)]     )

__global__
void cell_live(const unsigned char* const in_world,
                     unsigned char* const out_world,
                             unsigned int numRows,
                             unsigned int numCols)
{
    const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                          blockIdx.y * blockDim.y + threadIdx.y);
    // Check for out of boundries
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
      return;

    const unsigned int thread_1D_pos  = thread_2D_pos.y * numCols + thread_2D_pos.x;
    //debug---------------------------------
      //if(in_world[thread_1D_pos] != 0 && in_world[thread_1D_pos] != 255)
        //printf("non binary: %d\n",in_world[thread_1D_pos]);
    //debug---------------------------------


    unsigned int alive_neighbors = live_cells(thread_1D_pos, numCols, numRows, in_world);
    alive_neighbors /= 255;
    if (alive_neighbors == 3 || (alive_neighbors == 2 && in_world[thread_1D_pos]))
        out_world[thread_1D_pos] = 255;
    else
        out_world[thread_1D_pos] = 0;

}

void game_of_life_cuda(unsigned char* const d_inWorld,
                       unsigned char* const d_outWorld,
                       const size_t numRows,
                       const size_t numCols)
{
    const dim3 blockSize ( 32,32 );

                              // Ceiling
    const dim3 gridSize ( 1 + ((numCols - 1) / blockSize.x),   1 + ((numRows - 1) / blockSize.y) );
    //const dim3 gridSize ( numCols,numRows,1 );
    // Launch a kernel
    cell_live <<<gridSize, blockSize>>>(d_inWorld, d_outWorld, numRows, numCols);

    // Make sure I didn't did not mess this up like I did with my last relationship
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
