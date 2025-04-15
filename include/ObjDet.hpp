#ifndef OBJDET_NODE_H
#define OBJDET_NODE_H

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

class ObjectDetection : public rclcpp::Node {
    public:
        ObjectDetection ();
    private:
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription;
        pcl::visualization::PCLVisualizer::Ptr viewer;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        void PointCloudReceivedCallback (const sensor_msgs::msg::PointCloud2 & msg);
        void Start ();
        void Stop ();
        void DrawVector (Eigen::Vector3f vector, pcl::PointXYZ offset, float length, uint8_t r, uint8_t g, uint8_t b);
        void DrawPlane (Eigen::Vector4f& planeParameters);
        void RemovePoints (std::vector<int> indicesToRemove);
        void RemoveFarPoints (float threshold);
        void RemoveClosePoints (float threshold);
        Eigen::Vector3f NormalOfPlaneCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, std::vector<int> indices);
        Eigen::Vector4f CalculateFloorNormal (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
};

#endif
