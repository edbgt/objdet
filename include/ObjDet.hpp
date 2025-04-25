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
        std::vector<pcl::RGB> palette;
        Eigen::Vector4f floorParams;
        Eigen::Vector3f floorNormal;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription;
        pcl::visualization::PCLVisualizer::Ptr viewer;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, filteredCloud;
        pcl::SampleConsensusModelParallelPlane<pcl::PointXYZ>::Ptr parallelPlaneModel;
        pcl::RandomSampleConsensus<pcl::PointXYZ>::Ptr parallelPlaneRansac;

        uint8_t CreatePalette ();
        void DetectRemoveFloor (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
        std::vector<pcl::PointIndices> CreateClusters (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
        void ColorClusters (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, const std::vector<pcl::PointIndices> & clusterIndices);
        void PointCloudReceivedCallback (const sensor_msgs::msg::PointCloud2 & msg);
        void Downsample (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud, float leafSize);
        void Start ();
        void Stop ();
        void DrawVector (Eigen::Vector3f vector, pcl::PointXYZ offset, float length, uint8_t r, uint8_t g, uint8_t b);
        void DrawPlane (const Eigen::Vector4f& planeParameters, const std::string& id);
        void RemovePoints (std::vector<int> indicesToRemove);
        void RemoveFarPoints (float threshold);
        void RemoveClosePoints (float threshold);
        Eigen::Vector3f NormalOfPlaneCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, std::vector<int> indices);
        /*
         * Detect the biggest plane in the point cloud which is assumed to be the floor.
         *
         * @param inputCloud    cloud where floor gets detected
         * @param remove        set to true to remove the floor points from input cloud
         * @return              vector containing x y z w values of floor plane
         */
        Eigen::Vector4f CalculateFloorNormal (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, bool remove);
        /**
         * Find a plane that is orthogonal to the reference plane.
         *
         * @param normal        normal vector of reference plane
         * @param epsAngle      TODO
         * @param distThreshold maximum distance of a point from the plane
         * @return              indices of points belonging to plane
         */
        std::vector<int> FindOrthogonalPlaneRansac (Eigen::Vector3f normal, float epsAngle, float distThreshold);
        float MaxExtent (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud);
        void EstimateNormalsBoundaries ();
};

#endif
