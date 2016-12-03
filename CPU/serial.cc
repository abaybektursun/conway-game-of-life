#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define ITERS 100

std::vector<uchar> world;

// Write about infinite world and circular index

#define live_cells( x, y, xN, yN,world ) (    world[abs((x-1)%xN)][abs((y-1)%yN)] + world[x][abs((y-1)%yN)] + world[abs((x+1)%xN)][abs((y-1)%yN)]     \
                                   +          world[abs((x-1)%xN)][y]        +                                world[abs((x+1)%xN)][y]                 \
                                   +          world[abs((x-1)%xN)][abs((y+1)%yN)] + world[x][abs((y+1)%yN)] + world[abs((x+1)%xN)][abs((y+1)%yN)]     )  

int main()
{    
    //----------------------------------------------------------------------------------------------
    // Read user's image
    cv::Mat userImg = cv::imread("imgs/0.png",0);
    cv::Mat uImg8UC = cv::Mat(cv::Size(userImg.rows, userImg.cols),CV_8UC1);
    // Convert the user image to black and white using threshold
    cv::threshold( userImg, uImg8UC, 128, 255, CV_THRESH_BINARY );
    cv::imwrite( "imgs/1.png",uImg8UC ); 
    //----------------------------------------------------------------------------------------------
    // DEBUG ##################################################
    //printf("userImg: %d :  %d \n", userImg.rows, userImg.cols);
    //printf("uImg8UC: %d :  %d \n", uImg8UC.rows, uImg8UC.cols);
    // DEBUG ##################################################
    
    // Convert the matrix into vector
    std::vector<uchar> world1D;
    if (uImg8UC.isContinuous()) 
    {
        world1D.assign(uImg8UC.datastart, uImg8UC.dataend);
    } 
    else 
    {
        for (int i = 0; i < uImg8UC.rows; ++i) {
            world1D.insert(world1D.end(), uImg8UC.ptr<uchar>(i), uImg8UC.ptr<uchar>(i)+uImg8UC.cols);
        }
    }
    
    
    // DEBUG ##################################################
    //printf("world1D: %d \n", world1D.size());
    // DEBUG ##################################################
    
    // 1D to 2D
    std::vector<std::vector<uchar>> world;
    world.resize(uImg8UC.rows);
    for (int i = 0; i < uImg8UC.rows; i++)
    {
        world[i].resize(uImg8UC.cols);
    }
    for (unsigned i = 0; i < world1D.size(); i++)
    {
        int row = i / uImg8UC.cols;
        int col = i % uImg8UC.cols;
        world[row][col] = world1D[i];
    }
    
    // DEBUG ##################################################
    //printf("world: %d :  %d \n", world.size(), world[0].size());
    // DEBUG ##################################################
    
    std::srand(time(NULL));
    // Game
    for(int iter = 2; iter<ITERS; iter++)
    {
        std::vector<std::vector<uchar>> world_save;
        world_save.resize(uImg8UC.rows);
        for (int a_row = 0; a_row < uImg8UC.rows; ++a_row)
            world_save[a_row].resize(uImg8UC.cols);

        for (int row = 0; row < uImg8UC.rows; row++)
        {
            for(int col = 0; col < uImg8UC.cols; col++)
            {
                int alive_neighbors = live_cells(row,col,uImg8UC.rows,uImg8UC.cols, world);
                alive_neighbors /= 255;
                
                if (alive_neighbors == 3 || (alive_neighbors == 2 && world[row][col]))
                    world_save[row][col] = 255;
                else 
                    world_save[row][col] = 0;
            }
        }
        
        world = world_save;
        // DEBUG ##################################################
        //printf("\t after game: \n");
        //printf("world: %d :  %d \n", world.size(), world[0].size());
        // DEBUG ##################################################
        
        for(int row = 0; row <  uImg8UC.rows; row++)
            for(int col = 0; col <  uImg8UC.cols; col++)
                world1D[(row*uImg8UC.cols) + col] = world_save[row][col];
            
        // DEBUG ##################################################
        //printf("\t  2d to 1d: \n");
        //printf("world1D: %d \n", world1D.size());
        // DEBUG ##################################################
        
        cv::Mat a_frame = cv::Mat(cv::Size(uImg8UC.cols, uImg8UC.rows),CV_8UC1);
        
        // DEBUG ##################################################
        //printf("\t new frame: \n");
        //printf("a_frame: %d :  %d \n", a_frame.rows, a_frame.cols);
        // DEBUG ##################################################
        
        memcpy(a_frame.data, world1D.data(), world1D.size()*sizeof(uchar));
        
        // DEBUG ##################################################
        //printf("\t memcpy: \n");
        //printf("a_frame: %d :  %d \n", a_frame.rows, a_frame.cols);
        // DEBUG ##################################################
        
        std::string fileN = "imgs/" + std::to_string(iter) + ".png";
    
        
        try {
            cv::imwrite( fileN,a_frame );
        }
        catch (std::runtime_error& ex) {
            fprintf(stderr, "Exception converting matrix to image: %s\n", ex.what());
            return 1;
        }
    }
    
    return 0;
}


