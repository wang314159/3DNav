#include<iostream>
#include<pcl/io/ply_io.h>  // ply 文件读取头文件
#include<pcl/visualization/cloud_viewer.h>
#include <unistd.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>//点云可视化
#include <boost/thread/thread.hpp>//多线程
#include<math.h>
#include<opencv2/opencv.hpp>

#define MAP_SIZE  10.0
#define GRID_NUM  10
#define Z_OFFSET -0.5
#define IMAGE_SIZE  400
#define RATIO 100
#define FILTER_TIMES 10

class elevation{
public:
    
    double Grid[GRID_NUM][GRID_NUM];
    double Grid_copy[GRID_NUM][GRID_NUM];
    int num[GRID_NUM][GRID_NUM];
    std::vector<double> heightvec;
    float max;
    float min;
    elevation(){heightvec.resize(GRID_NUM*GRID_NUM);};
    void Input(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud);
    std::vector<double>& GetHeightVec();
    void Filter();
};

int main(int argc, char** argv) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPLYFile("map.ply", *pointCloud);
    cv::Mat img(IMAGE_SIZE, IMAGE_SIZE, CV_8UC4, cv::Scalar(0, 0, 0));
    int image_grid = IMAGE_SIZE/GRID_NUM;
    elevation ele;
    ele.Input(pointCloud);
    std::cout<<ele.min<<"   "<<ele.max<<std::endl;
    for(int i =0; i<GRID_NUM*GRID_NUM;i++){
        std::cout<<ele.GetHeightVec()[i]<<" ";
        std::cout<<ele.Grid[i/GRID_NUM][i%GRID_NUM]<<std::endl;
    }
    for(int i = 0 ;i<GRID_NUM ; i++){
        for(int j=0;j<GRID_NUM;j++){
            double height = ele.Grid[j][i];
            if(!finite(height)) height = 1;
            int scalar = (int)((height-ele.min+1)/(ele.max-ele.min+2)*255);
            cv::rectangle(img, cv::Point(image_grid*i, image_grid*j), cv::Point(image_grid*(i+1), image_grid*(j+1)), 
            cv::Scalar( scalar,  scalar, scalar), -1);
        }
        // std::cout<<std::endl;
    }
    pcl::visualization::PCLVisualizer viewer; 
    viewer.addPointCloud(pointCloud);
    cv::imshow("image", img);
    while (!viewer.wasStopped())
	{
        
		viewer.spinOnce(100);
	}
    cv::waitKey(0);
    return 0;
}

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
