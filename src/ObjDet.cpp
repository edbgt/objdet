#include <functional>
#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <pcl/common/common.h>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "ObjDet.hpp"

ObjectDetection::ObjectDetection () : rclcpp::Node ("ObjectDetectionNode") {
    subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>("tof_point_cloud", 10, std::bind(&ObjectDetection::PointCloudReceivedCallback, this, std::placeholders::_1));
    rclcpp::on_shutdown(std::bind(&ObjectDetection::Stop, this));
    this->Start();
}

void ObjectDetection::PointCloudReceivedCallback (const sensor_msgs::msg::PointCloud2 & msg) {
    RCLCPP_INFO(get_logger(), "received point cloud of size %u * %u", msg.width, msg.height);
    // convert msg to pc
    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
    pcl::fromROSMsg(msg, *(this->cloud));
    RCLCPP_DEBUG(get_logger(), "converted");
    // show in viewer
    this->viewer->removeAllPointClouds();
    this->viewer->addPointCloud<pcl::PointXYZ>(this->cloud, "received cloud", 0);
    this->viewer->spinOnce();
    RCLCPP_DEBUG(get_logger(), "added");
}

void ObjectDetection::Start () {
    RCLCPP_INFO(get_logger(), "starting");
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
    this->viewer.reset(new pcl::visualization::PCLVisualizer ());
    this->viewer->setBackgroundColor(0.0, 0.0, 0.0);
    this->viewer->setCameraPosition(-4.60736, 0.725677, 0.738424, 0.0829958, 0.963966, -0.252749);
    this->viewer->addCoordinateSystem();
    this->viewer->setShowFPS(true);
}

void ObjectDetection::Stop () {
    RCLCPP_INFO(get_logger(), "shutting down");
}

int main (int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetection>());
    rclcpp::shutdown();
    return 0;
}
