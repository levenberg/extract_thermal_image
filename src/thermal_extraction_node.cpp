
#include <cstdio>
#include <vector>
#include <iomanip>
#include <fstream>
#include <ros/ros.h>
#include <omp.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include "file_io.hpp"

std::string thermalDir;
struct Data
{
    Data(FILE *f)
    {
        if (fscanf(f, " %lf,%d,%lf", &t_b, &seq, &t_h) != EOF)
        {
            ;
        }
    }
    double t_b, t_h;
    int seq;
};

std::vector<Data> selected_timestamps;
int idx=0;
double pre_timestamp =0.0;
void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    if (image_msg->header.stamp.toSec() > selected_timestamps.back().t_h+1.0)
      return;
    // std::cout << "rgb seq: " << selected_timestamps[idx].seq << ", rgb t:" << selected_timestamps[idx].t_h << ", ir t:" << image_msg->header.stamp.toSec() << std::endl;
    if (selected_timestamps[idx].t_h >= pre_timestamp && selected_timestamps[idx].t_h <= image_msg->header.stamp.toSec())
    {
        cv_bridge::CvImageConstPtr ptr;
        ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::TYPE_8UC1);
        cv::Mat image = ptr->image;
        char thermal_file[1000];
        const char *cstr = thermalDir.c_str();
        std::sprintf(thermal_file, "%s/%.6d.png", cstr, selected_timestamps[idx].seq);
        std::cout << "image id: " << selected_timestamps[idx].seq << ", rgb t:" << selected_timestamps[idx].t_h << ", ir t:" << image_msg->header.stamp.toSec() << std::endl;
        cv::imwrite(thermal_file, image);
        idx++;

        // write result to file
        std::string thermal_filelist = thermalDir+"/thermal_filelist.csv";
        std::ofstream foutC(thermal_filelist, std::ios::app);
        foutC << thermal_file<<std::endl;
        foutC.close();
    }
    pre_timestamp = image_msg->header.stamp.toSec();
    
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "thermal_extraction_node");
    ros::NodeHandle n("~");

    std::string timestamps_csv, rgbimage_csv, thermal_topic;
    n.param("timestamps_csv",   timestamps_csv,   std::string(""));
    n.param("rgbimage_csv",   rgbimage_csv,   std::string(""));
    n.param("thermal_dir",   thermalDir,   std::string(""));
    n.param("thermal_topic",   thermal_topic,   std::string(""));

    std::vector<Data> rgb_timestamps;
    FILE *f = fopen(timestamps_csv.c_str(), "r");
    if (f==NULL)
    {
      ROS_WARN("can't load timestamps; wrong path");
      return 0;
    }
    char tmp[10000];
    if (fgets(tmp, 10000, f) == NULL)
    {
        ROS_WARN("can't load timestamps; no data available");
    }
    while (!feof(f))
        rgb_timestamps.emplace_back(f);
    fclose(f);
    rgb_timestamps.pop_back();
    ROS_INFO("Data loaded: %d, %lf, %d, %lf", (int)rgb_timestamps.size(), rgb_timestamps[0].t_b, rgb_timestamps[0].seq, rgb_timestamps[0].t_h);

    std::vector<std::string> ImageFileNames;
    readData(rgbimage_csv, ImageFileNames);
    size_t nImages = ImageFileNames.size();
    if(nImages == 0) {
        std::cerr << "No image files (.png or .jpg) are found in given folder. " << std::endl;
        return 1;
    }

    // Pick the timestamp of images in the full cvs
    
    for (size_t i=0; i<nImages; i++)
    {
        std::string name=ImageFileNames[i];
        int len = name.length();
        int image_index = std::stoi(name.substr(len-10,len-4));
        ROS_INFO("image index: %d", image_index);

        Data select = rgb_timestamps[image_index];
        select.seq = image_index;
        selected_timestamps.push_back(select);
    }
    std::cout<<"first time: "<<std::setprecision(20)<<selected_timestamps.back().t_h<<std::endl;
    ros::Subscriber sub_thermal_image = n.subscribe(thermal_topic, 1000, image_callback);
    
    ros::Rate r(100);
    ros::spin();


    std::cout << "thermal intensity projection finished, time: "<< std::endl;

    return 0;
}
