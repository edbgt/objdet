#include <functional>
#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>

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
    pcl::fromROSMsg(msg, *(this->cloud));
    // show in viewer
    this->viewer->removeAllPointClouds();
    this->viewer->addPointCloud<pcl::PointXYZ>(this->cloud, "received cloud", 0);
    // detect floor plane
    Eigen::Vector4f floorParameters = CalculateFloorNormal(this->cloud);
    DrawPlane(floorParameters);
    // reduce number of points in cloud
    RemoveFarPoints(1.0); // m
    RemoveClosePoints(0.5); // m
    // setup parallel plane model
    Eigen::Vector3f floorNormal = {floorParameters.x(), floorParameters.y(), floorParameters.z()};
    this->parallelPlaneModel->setInputCloud(this->cloud);
    this->parallelPlaneModel->setAxis(floorNormal);
    this->parallelPlaneModel->setEpsAngle(pcl::deg2rad(0.5f));
    // setup parallel plane ransac
    this->parallelPlaneRansac->setDistanceThreshold(0.005);
    // get first plane
    this->parallelPlaneRansac->computeModel();
    std::vector<int> firstPlaneIndices;
    this->parallelPlaneRansac->getInliers(firstPlaneIndices);
    pcl::RGB firstRgb (255, 0, 255);
    pcl::PointCloud<pcl::PointXYZ>::Ptr first (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> firstColor (first, firstRgb.r, firstRgb.g, firstRgb.b);
    pcl::copyPointCloud(*(this->cloud), firstPlaneIndices, *first);
    if (!this->viewer->addPointCloud<pcl::PointXYZ> (first, firstColor, "first cuboid side")) {
        RCLCPP_ERROR(get_logger(), "could not add first cuboid side point cloud");
    }
    RemovePoints(firstPlaneIndices);
    this->viewer->spinOnce();
}

void ObjectDetection::Start () {
    RCLCPP_INFO(get_logger(), "starting");
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ> ());
    this->viewer.reset(new pcl::visualization::PCLVisualizer ());
    RCLCPP_DEBUG(get_logger(), "initializing parallel plane model");
    this->parallelPlaneModel.reset(new pcl::SampleConsensusModelParallelPlane<pcl::PointXYZ> (this->cloud));
    RCLCPP_DEBUG(get_logger(), "initializing parallel plane ransac");
    this->parallelPlaneRansac.reset(new pcl::RandomSampleConsensus<pcl::PointXYZ> (this->parallelPlaneModel));
    this->viewer->setBackgroundColor(0.0, 0.0, 0.0);
    this->viewer->setCameraPosition(-4.60736, 0.725677, 0.738424, 0.0829958, 0.963966, -0.252749);
    this->viewer->addCoordinateSystem();
    this->viewer->setShowFPS(true);
}

void ObjectDetection::Stop () {
    RCLCPP_INFO(get_logger(), "shutting down");
}

void ObjectDetection::DrawVector (Eigen::Vector3f vector, pcl::PointXYZ offset, float length, uint8_t r, uint8_t g, uint8_t b) {
    pcl::PointXYZ start = {offset.x, offset.y, offset.z};
    pcl::PointXYZ end = {length * vector.x() + offset.x, length * vector.y() + offset.y, length * vector.z() + offset.z};
    std::stringstream id;
    id << std::to_string(r) << std::to_string(g) << std::to_string(b);
    if (!this->viewer->addArrow(start, end, r, g, b, false, id.str())) {
        RCLCPP_ERROR(get_logger(), "could not draw arrow");
    }
}

void ObjectDetection::DrawPlane (Eigen::Vector4f& planeParameters) {
    pcl::ModelCoefficients coefficients;
    coefficients.values.resize(4);
    coefficients.values[0] = planeParameters.x();
    coefficients.values[1] = planeParameters.y();
    coefficients.values[2] = planeParameters.z();
    coefficients.values[3] = planeParameters.w();
    if (!this->viewer->addPlane(coefficients)) {
        RCLCPP_ERROR(get_logger(), "could not draw plane");
    }
}

void ObjectDetection::RemovePoints (std::vector<int> indicesToRemove) {
    // could maybe be improved, see documentation: https://pointclouds.org/documentation/classpcl_1_1_extract_indices.html#add1af519a1a4d4d2665e07a942262aac
    pcl::IndicesPtr indicesPtr = std::make_shared<std::vector<int>> (indicesToRemove);
    pcl::ExtractIndices<pcl::PointXYZ> extractIndicesFilter;
    extractIndicesFilter.setInputCloud(this->cloud);
    extractIndicesFilter.setIndices(indicesPtr);
    extractIndicesFilter.setNegative(true);
    extractIndicesFilter.filter(*(this->cloud));
}

void ObjectDetection::RemoveFarPoints (float threshold) {
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu before removing far points", this->cloud->size());
    std::vector<int> farIndices;
    for (int i = 0; i != this->cloud->size(); ++i) {
        if (this->cloud->points.at(i).z > threshold) {
            farIndices.push_back(i);
        }
    }
    this->RemovePoints(farIndices);
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu after removing far points", this->cloud->size());
}

void ObjectDetection::RemoveClosePoints (float threshold) {
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu before removing close points", this->cloud->size());
    std::vector<int> closeIndices;
    for (int i = 0; i != this->cloud->size(); ++i) {
        if (this->cloud->points.at(i).z < threshold) {
            closeIndices.push_back(i);
        }
    }
    this->RemovePoints(closeIndices);
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu after removing close points", this->cloud->size());
}

Eigen::Vector3f ObjectDetection::NormalOfPlaneCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, std::vector<int> indices) {
    // maybe lose first parameter and operate only on indices?
    Eigen::Vector4f planeParameters;
    float curve;
    pcl::computePointNormal(*cloud, indices, planeParameters, curve);
    Eigen::Vector3f normal = {planeParameters.x(), planeParameters.y(), planeParameters.z()};
    return normal;
}

Eigen::Vector4f ObjectDetection::CalculateFloorNormal (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud) {
    // biggest plane should be floor
    std::vector<int> inlierIndices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr floorCloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr planeModel (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (inputCloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (planeModel);
    ransac.setDistanceThreshold(0.02);
    ransac.computeModel();
    ransac.getInliers(inlierIndices);
    Eigen::Vector4f floorParameters;
    float curve;
    pcl::computePointNormal(*inputCloud, inlierIndices, floorParameters, curve);
    return floorParameters;
}

int main (int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetection>());
    rclcpp::shutdown();
    return 0;
}
