#ifndef ELEVATIONMAP_HPP
#define ELEVATIONMAP_HPP

#include<iostream>
#include <unistd.h>
#include <pcl/common/common_headers.h>
#include<math.h>
#include<opencv2/opencv.hpp>

#define MAP_SIZE  10.0
#define GRID_NUM  20
#define Z_OFFSET -0.5
#define IMAGE_SIZE  400
#define RATIO 100
#define FILTER_TIMES 10

class elevation{
private:
double Grid[GRID_NUM][GRID_NUM];
    double Grid_copy[GRID_NUM][GRID_NUM];
    int num[GRID_NUM][GRID_NUM];
    std::vector<double> heightvec;
    float max;
    float min;
public:
    elevation(){heightvec.resize(GRID_NUM*GRID_NUM);};
    void Input(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud);
    std::vector<double>& GetHeightVec();
    void Filter();
};

void elevation::Filter(){
    memcpy(Grid_copy,Grid,GRID_NUM*GRID_NUM*sizeof(double));
    // std::copy(Grid,Grid+GRID_NUM*GRID_NUM,Grid_copy);
    for(int i = 0 ;i<GRID_NUM ; i++){
        for(int j=0;j<GRID_NUM;j++){
            double val=Grid_copy[i][j];
            double mean_x=0,mean_y=0;
            if(i>0&&i<GRID_NUM-1){
                mean_x = (Grid_copy[i+1][j]+Grid_copy[i-1][j])/2;
                // val=abs(Grid_copy[i][j])>abs(mean)?Grid_copy[i][j]:mean;
            }
            if(j>0&&j<GRID_NUM-1){
                mean_y = (Grid_copy[i][j+1]+Grid_copy[i][j-1])/2;
            }
            if(abs(mean_x)>abs(val)||abs(mean_y)>abs(val))
                val=abs(mean_x)>abs(mean_y)?mean_x:mean_y;
            if (abs(val)<1||(!finite(val))) val=1;
            if (abs(val)>5000) val=50;

            Grid[i][j]=val;
        }
        // std::cout<<std::endl;
    }
}

void elevation::Input(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud)
{
    for( int i=0 ; i < pointCloud->points.size();i++){
        float x = pointCloud->points[i].x;
        float y = pointCloud->points[i].y;
        if(abs(x)<MAP_SIZE/2 && abs(y)<MAP_SIZE/2){
            float z = pointCloud->points[i].z*RATIO;
            max = z>max?z:max;
            min = z<min?z:min;
            int x_index = GRID_NUM/2 + floor(x/MAP_SIZE*GRID_NUM);
            int y_index = GRID_NUM/2 + floor(y/MAP_SIZE*GRID_NUM);
            int n = num[x_index][y_index]++;
            Grid[x_index][y_index] = Grid[x_index][y_index]/(n+1)*n + z/(n+1);
        }
    }

    for(int i=0;i<FILTER_TIMES;i++){
        Filter();
    }
    memcpy(&heightvec[0],Grid,GRID_NUM*GRID_NUM*sizeof(double));
}


std::vector<double>& elevation::GetHeightVec()
{
    
    return heightvec;
}

#endif