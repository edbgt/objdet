#include <functional>
#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <pcl/common/common.h>
#include <pcl_ros/transforms.hpp>

#include "ObjDet.hpp"

ObjectDetection::ObjectDetection () : rclcpp::Node ("ObjectDetectionNode") {
    subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>("tof_point_cloud", 10, std::bind(&ObjectDetection::PointCloudReceivedCallback, this, std::placeholders::_1));
    rclcpp::on_shutdown(std::bind(&ObjectDetection::Stop, this));
    this->Start();
}

void ObjectDetection::PointCloudReceivedCallback (const sensor_msgs::msg::PointCloud2 & msg) {
    RCLCPP_INFO(get_logger(), "received point cloud of size %u * %u", msg.width, msg.height);
    // show in viewer
}

void ObjectDetection::Start () {
    RCLCPP_INFO(get_logger(), "starting");
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
