#include <ros/ros.h>
#include <omp.h>
#include <Eigen/Core>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

ros::Publisher pub_img;
std::string thermal_topic;
double max_intensity, min_intensity;
bool auto_thresh=false;

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
  cv_bridge::CvImageConstPtr ptr;
  ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::TYPE_16UC1);
  cv::Mat IR16 = ptr->image;
  if (auto_thresh)
    cv::minMaxIdx(IR16, &min_intensity, &max_intensity);
  cv::Mat IR8(IR16.rows, IR16.cols, 0);
  for (int i = 0; i < IR8.rows * IR8.cols; i++)
  {
    auto pixel= ((uint16_t *)ptr->image.data)[i];
    pixel = pixel>max_intensity?max_intensity:pixel;
    pixel = pixel<min_intensity?min_intensity:pixel;
    IR8.data[i] = uchar((pixel-min_intensity)* 255.0/(max_intensity-min_intensity));
  }
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(image_msg->header, "8UC1", IR8).toImageMsg();
  pub_img.publish(msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "IR 16 bit image processing");
    ros::NodeHandle n("~");

    n.param("thermal_topic",   thermal_topic,   std::string(""));
    n.getParam("auto_thresh", auto_thresh);
    n.getParam("min_intensity", min_intensity);
    n.getParam("max_intensity", max_intensity);
    pub_img = n.advertise<sensor_msgs::Image>("/thermal/image8",1000);

    ros::Subscriber sub_image = n.subscribe(thermal_topic, 1000, image_callback);

    // ros::Rate r(20);
    ros::spin();
}