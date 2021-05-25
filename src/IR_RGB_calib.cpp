#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

std::queue<sensor_msgs::ImageConstPtr> rgb_buf;
std::queue<sensor_msgs::ImageConstPtr> thermal_buf;
std::mutex m_buf;
std::string rgbDir, thermalDir;
void thermal_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    // ROS_INFO("thermal received");
    m_buf.lock();
    thermal_buf.push(image_msg);
    m_buf.unlock();
}
void rgb_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    // ROS_INFO("rgb received");
    m_buf.lock();
    rgb_buf.push(image_msg);
    m_buf.unlock();
}
void process()
{
    unsigned int idx=0;
    while (true)
    {
        sensor_msgs::ImageConstPtr rgb_msg = NULL;
        sensor_msgs::ImageConstPtr thermal_msg = NULL;
        if (!rgb_buf.empty() && !thermal_buf.empty())
        {
            if (rgb_buf.front()->header.stamp.toSec() > thermal_buf.front()->header.stamp.toSec())
            {
                thermal_buf.pop();
                // printf("throw pose at beginning\n");
            }
            else
            {
                rgb_msg = rgb_buf.front();
                rgb_buf.pop();

                // while (thermal_buf.front()->header.stamp.toSec() > rgb_msg->header.stamp.toSec())
                // {
                    // ROS_INFO("thermal_buf size: %d", thermal_buf.size());
                    thermal_msg = thermal_buf.front();
                    thermal_buf.pop();
                // }
            }
        }

        if(rgb_msg != NULL && thermal_msg!= NULL)
        {
            ROS_INFO("rgb time %lf, thermal time %lf", rgb_msg->header.stamp.toSec(), thermal_msg->header.stamp.toSec());
            cv_bridge::CvImageConstPtr ptr_thermal;
            ptr_thermal = cv_bridge::toCvCopy(thermal_msg, sensor_msgs::image_encodings::TYPE_8UC1);
            cv::Mat thermal = ptr_thermal->image;
            if(thermal.channels()==3)
                cv::cvtColor(thermal, thermal, CV_RGB2GRAY);
            std::string thermal_file = thermalDir + std::to_string(idx) + ".jpg";
            cv::imwrite(thermal_file, thermal);

            cv_bridge::CvImageConstPtr ptr_rgb;
            ptr_rgb = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BAYER_BGGR8);
            cv::Mat rgb = ptr_rgb->image;
            cv::Size dsz= thermal.size();
            cv::Mat new_rgb;
            cv::resize(rgb, new_rgb, dsz);
            if(rgb.channels()==3)
                cv::cvtColor(rgb, rgb, CV_BayerRG2GRAY);
            std::string rgb_file = rgbDir + std::to_string(idx) + ".jpg";
            cv::imwrite(rgb_file, new_rgb);
            idx++;
        }
    }
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "thermal-rgb");
    ros::NodeHandle n("~");

    std::string rgb_topic, thermal_topic;
    n.param("rgbDir",   rgbDir,   std::string(""));
    n.param("thermalDir",   thermalDir,   std::string(""));
    n.param("rgb_topic",   rgb_topic,   std::string(""));
    n.param("thermal_topic",   thermal_topic,   std::string(""));

    ros::Subscriber sub_thermal_image = n.subscribe(thermal_topic, 1000, thermal_callback);
    ros::Subscriber sub_rgb_image = n.subscribe(rgb_topic, 1000, rgb_callback);
    
    ros::Rate r(100);
    std::thread joint_process;
    joint_process = std::thread(process);
    ros::spin();
    
    return 0;
}
