#include "utils.h"


// Mod. operation for preventing out of boundry index
#define live_cells( x, y, xN, yN,world ) (    world[abs((x-1)%xN)][abs((y-1)%yN)] + world[x][abs((y-1)%yN)] + world[abs((x+1)%xN)][abs((y-1)%yN)]     \
                                   +          world[abs((x-1)%xN)][y]        +                                world[abs((x+1)%xN)][y]                 \
                                   +          world[abs((x-1)%xN)][abs((y+1)%yN)] + world[x][abs((y+1)%yN)] + world[abs((x+1)%xN)][abs((y+1)%yN)]     )

__global__
void cell_live(uchar1** world)
{

}

void game_of_life_cuda(const uchar1* const h_inWorld,
                             uchar1* const d_inWorld,
                             uchar1* const d_outWorld,
                       const size_t numRows,
                       const size_t numCols)
{

}

void memoryManagement(const size_t numRows, const size_t numCols)
{

}
