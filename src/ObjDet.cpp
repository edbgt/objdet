#include <functional>
#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "ObjDet.hpp"

uint8_t ObjectDetection::CreatePalette () {
    this->palette.emplace_back(0x2f, 0x4f, 0x4f); // darkslategray
    this->palette.emplace_back(0x6b, 0x8e, 0x23); // olivedrab
    this->palette.emplace_back(0xa0, 0x52, 0x2d); // sienna
    this->palette.emplace_back(0x19, 0x19, 0x70); // midnightblue
    this->palette.emplace_back(0xff, 0x00, 0x00); // red
    this->palette.emplace_back(0xff, 0xa5, 0x00); // orange
    this->palette.emplace_back(0x00, 0x00, 0xcd); // mediumblue
    this->palette.emplace_back(0x7f, 0xff, 0x00); // chartreuse
    this->palette.emplace_back(0x00, 0xfa, 0x9a); // mediumspringgreen
    this->palette.emplace_back(0x00, 0xff, 0xff); // aqua
    this->palette.emplace_back(0xff, 0x00, 0xff); // fuchsia
    this->palette.emplace_back(0x1e, 0x90, 0xff); // dodgerblue
    this->palette.emplace_back(0xff, 0xff, 0x54); // laselemon
    this->palette.emplace_back(0xdd, 0xa0, 0xdd); // plum
    this->palette.emplace_back(0xff, 0x14, 0x93); // deeppink
    this->palette.emplace_back(0xf5, 0xde, 0xb3); // wheat
    return 16;
}

ObjectDetection::ObjectDetection () : rclcpp::Node ("ObjectDetectionNode") {
    subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>("tof_point_cloud", 1, std::bind(&ObjectDetection::PointCloudReceivedCallback, this, std::placeholders::_1));
    rclcpp::on_shutdown(std::bind(&ObjectDetection::Stop, this));
    this->Start();
}

void ObjectDetection::PointCloudReceivedCallback (const sensor_msgs::msg::PointCloud2 & msg) {
    RCLCPP_DEBUG(get_logger(), "received point cloud of size %u * %u", msg.width, msg.height);
    // convert msg to pc
    pcl::fromROSMsg(msg, *(this->cloud));
    // remove point clouds from viewer
    this->viewer->removeAllPointClouds();
    // show point cloud
    //this->viewer->addPointCloud<pcl::PointXYZ>(this->cloud, "received cloud", 0);
    //DetectRemoveFloor(this->cloud);
    this->floorParams = CalculateFloorNormal(this->cloud, true);
    this->floorNormal = {floorParams.x(), floorParams.y(), floorParams.z()};
    // remove far and close points in cloud
    RemoveFarPoints(this->cloud, 0.6); // m
    RemoveClosePoints(this->cloud, 0.2); // m
    // extract clusters
    std::vector<pcl::PointIndices> clusterIndices = CreateClusters(this->cloud);
    //ColorClusters(this->cloud, clusterIndices);
    FindCuboidCluster(this->cloud, clusterIndices);
    // update viewer
    this->viewer->spinOnce();
}

void ObjectDetection::Downsample (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud, float leafSize) {
    RCLCPP_DEBUG(get_logger(), "size before downsampling: %lu", inputCloud->size());
    pcl::VoxelGrid<pcl::PointXYZ> voxelGrid;
    voxelGrid.setInputCloud(inputCloud);
    voxelGrid.setLeafSize(leafSize, leafSize, leafSize);
    voxelGrid.filter(*(outputCloud));
    RCLCPP_DEBUG(get_logger(), "size after downsampling: %lu", outputCloud->size());
}

void ObjectDetection::DetectRemoveFloor (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud) {
    pcl::SACSegmentation<pcl::PointXYZ> segmentation;
    pcl::PointIndices::Ptr inlierIndices (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr modelCoefficients (new pcl::ModelCoefficients);
    pcl::PointCloud<pcl::PointXYZ> tempCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr floorCloud (new pcl::PointCloud<pcl::PointXYZ> ());
    segmentation.setOptimizeCoefficients(true);
    segmentation.setModelType(pcl::SACMODEL_PLANE);
    segmentation.setMethodType(pcl::SAC_RANSAC);
    segmentation.setMaxIterations(100);
    segmentation.setDistanceThreshold(0.02);

    size_t initialCloudSize = inputCloud->size();
    while (inputCloud->size() > 0.3 * initialCloudSize) {
        segmentation.setInputCloud(inputCloud);
        segmentation.segment(*inlierIndices, *modelCoefficients);
        if (inlierIndices->indices.size() == 0) {
            RCLCPP_ERROR(get_logger(), "could not estimate planar model");
            return;
        }
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(inputCloud);
        extract.setIndices(inlierIndices);
        // put all points in inlierIndices into floorCloud
        extract.setNegative(false);
        extract.filter(*floorCloud);
        RCLCPP_INFO(get_logger(), "removing floor cloud with size %lu", floorCloud->size());
        // put all points not in inlierIndices in tempCloud and then in inputCloud
        extract.setNegative(true);
        extract.filter(tempCloud);
        *inputCloud = tempCloud;
    }
}

std::vector<pcl::PointIndices> ObjectDetection::CreateClusters (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud) {
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(inputCloud);
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> clusterExtraction;
    clusterExtraction.setClusterTolerance(0.02);
    clusterExtraction.setMinClusterSize(50);
    clusterExtraction.setMaxClusterSize(5000);
    clusterExtraction.setSearchMethod(tree);
    clusterExtraction.setInputCloud(inputCloud);
    clusterExtraction.extract(clusterIndices);
    RCLCPP_INFO(get_logger(), "extracted %lu clusters", clusterIndices.size());
    return clusterIndices;
}

void ObjectDetection::ColorClusters (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, const std::vector<pcl::PointIndices> & clusterIndices) {
    uint8_t counter = 0;
    for (const auto & cluster : clusterIndices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud (new pcl::PointCloud<pcl::PointXYZ> ());
        for (const auto & index : cluster.indices) {
            clusterCloud->push_back((*inputCloud)[index]);
        }
        clusterCloud->width = clusterCloud->size();
        clusterCloud->height = 1;
        clusterCloud->is_dense = true;
        std::stringstream ss;
        ss << "cluster" << std::setw(3) << std::setfill('0') << counter;
        RCLCPP_DEBUG(get_logger(), "drawing cluster with size %lu", clusterCloud->size());
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color (clusterCloud, this->palette.at(counter % 16).r, this->palette.at(counter % 16).g, this->palette.at(counter % 16).b);
        if (!this->viewer->addPointCloud<pcl::PointXYZ>(clusterCloud, color, ss.str().c_str())) {
            RCLCPP_ERROR(get_logger(), "could not draw cluster");
        }
        ++counter;
    }
}

void ObjectDetection::FindCuboidCluster (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, const std::vector<pcl::PointIndices> & clusterIndices) {
    uint8_t counter = 0;
    for (const auto & cluster : clusterIndices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud (new pcl::PointCloud<pcl::PointXYZ> ());
        for (const auto & index : cluster.indices) {
            clusterCloud->push_back((*inputCloud)[index]);
        }
        clusterCloud->width = clusterCloud->size();
        clusterCloud->height = 1;
        clusterCloud->is_dense = true;
        std::vector<int> firstSideIndices = FindOrthogonalPlaneRansac(clusterCloud, this->floorNormal, 1.0, 0.005);
        if (firstSideIndices.size()) {
            RCLCPP_DEBUG(get_logger(), "found first plane in cluster %hhu", counter);
            Eigen::Vector3f firstSideNormal = NormalOfPlaneCloud (clusterCloud, firstSideIndices);
            pcl::PointXYZ firstSideCentroid;
            pcl::computeCentroid(*clusterCloud, firstSideIndices, firstSideCentroid);
            DrawPointCloud(clusterCloud, firstSideIndices, 0);
            DrawSmallSphere(firstSideCentroid, 3);
            RemovePoints(clusterCloud, firstSideIndices);
            std::vector<int> secondSideIndices = FindOrthogonalPlaneRansac(clusterCloud, firstSideNormal, 1.0, 0.005);
            if (secondSideIndices.size()) {
                RCLCPP_DEBUG(get_logger(), "found second plane in cluster %hhu", counter);
                DrawPointCloud(clusterCloud, secondSideIndices, 1);
                pcl::PointXYZ secondSideCentroid;
                pcl::computeCentroid(*clusterCloud, secondSideIndices, secondSideCentroid);
                DrawSmallSphere(secondSideCentroid, 4);
            } else {
                RCLCPP_DEBUG(get_logger(), "could not find plane in cluster %hhu", counter);
            }
        } else {
            RCLCPP_DEBUG(get_logger(), "could not find plane in cluster %hhu", counter);
        }
        ++counter;
    }
}

void ObjectDetection::Start () {
    RCLCPP_INFO(get_logger(), "starting");
    CreatePalette();
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ> ());
    this->filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZ> ());
    this->viewer.reset(new pcl::visualization::PCLVisualizer ());
    RCLCPP_DEBUG(get_logger(), "initializing parallel plane model");
    this->parallelPlaneModel.reset(new pcl::SampleConsensusModelParallelPlane<pcl::PointXYZ> (this->cloud));
    RCLCPP_DEBUG(get_logger(), "initializing parallel plane ransac");
    this->parallelPlaneRansac.reset(new pcl::RandomSampleConsensus<pcl::PointXYZ> (this->parallelPlaneModel));
    this->viewer->setBackgroundColor(0.0, 0.0, 0.0);
    this->viewer->setCameraPosition(-3.12901, 0.151551, -0.847222, 0.0891162, 0.989655, -0.11243);
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

void ObjectDetection::DrawPlane (const Eigen::Vector4f& planeParameters, const std::string& id) {
    pcl::ModelCoefficients coefficients;
    coefficients.values.resize(4);
    coefficients.values[0] = planeParameters.x();
    coefficients.values[1] = planeParameters.y();
    coefficients.values[2] = planeParameters.z();
    coefficients.values[3] = planeParameters.w();
    if (!this->viewer->addPlane(coefficients, id)) {
        RCLCPP_ERROR(get_logger(), "could not draw plane");
    }
}

void ObjectDetection::DrawSmallSphere (pcl::PointXYZ center, uint8_t colorIndex) {
    std::stringstream id;
    id << std::to_string(this->palette.at(colorIndex).r) << std::to_string(this->palette.at(colorIndex).g) << std::to_string(this->palette.at(colorIndex).b);
    if (!this->viewer->addSphere(center, 0.005, this->palette.at(colorIndex).r, this->palette.at(colorIndex).g, this->palette.at(colorIndex).b, id.str())) {
        RCLCPP_ERROR(get_logger(), "could not draw sphere");
    }
}

void ObjectDetection::DrawPointCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, std::vector<int> indices, uint8_t colorIndex) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::copyPointCloud(*inputCloud, indices, *tempCloud);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color (tempCloud, this->palette.at(colorIndex).r, this->palette.at(colorIndex).g, this->palette.at(colorIndex).b);
    std::stringstream id;
    id << std::to_string(this->palette.at(colorIndex).r) << std::to_string(this->palette.at(colorIndex).g) << std::to_string(this->palette.at(colorIndex).b);
    if (!this->viewer->addPointCloud<pcl::PointXYZ>(tempCloud, color, id.str())) {
        RCLCPP_ERROR(get_logger(), "could not draw cloud with colorIndex %hhu", colorIndex);
    }
}

void ObjectDetection::RemovePoints (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, std::vector<int> indicesToRemove) {
    // could maybe be improved, see documentation: https://pointclouds.org/documentation/classpcl_1_1_extract_indices.html#add1af519a1a4d4d2665e07a942262aac
    pcl::IndicesPtr indicesPtr = std::make_shared<std::vector<int>> (indicesToRemove);
    pcl::ExtractIndices<pcl::PointXYZ> extractIndicesFilter;
    extractIndicesFilter.setInputCloud(inputCloud);
    extractIndicesFilter.setIndices(indicesPtr);
    extractIndicesFilter.setNegative(true);
    extractIndicesFilter.filter(*inputCloud);
}

void ObjectDetection::RemoveFarPoints (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, float threshold) {
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu before removing far points", inputCloud->size());
    std::vector<int> farIndices;
    for (int i = 0; i != inputCloud->size(); ++i) {
        if (inputCloud->points.at(i).z > threshold) {
            farIndices.push_back(i);
        }
    }
    this->RemovePoints(inputCloud, farIndices);
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu after removing far points", inputCloud->size());
}

void ObjectDetection::RemoveClosePoints (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, float threshold) {
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu before removing close points", inputCloud->size());
    std::vector<int> closeIndices;
    for (int i = 0; i != inputCloud->size(); ++i) {
        if (inputCloud->points.at(i).z < threshold) {
            closeIndices.push_back(i);
        }
    }
    this->RemovePoints(inputCloud, closeIndices);
    RCLCPP_DEBUG(get_logger(), "point cloud size is %lu after removing close points", inputCloud->size());
}

Eigen::Vector3f ObjectDetection::NormalOfPlaneCloud (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, std::vector<int> indices) {
    // maybe lose first parameter and operate only on indices?
    Eigen::Vector4f planeParameters;
    float curve;
    pcl::computePointNormal(*cloud, indices, planeParameters, curve);
    Eigen::Vector3f normal = {planeParameters.x(), planeParameters.y(), planeParameters.z()};
    return normal;
}

Eigen::Vector4f ObjectDetection::CalculateFloorNormal (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, bool remove) {
    // biggest plane should be floor
    std::vector<int> inlierIndices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr floorCloud (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr planeModel (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (inputCloud));
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (planeModel);
    ransac.setDistanceThreshold(0.01); // m
    ransac.computeModel();
    ransac.getInliers(inlierIndices);
    if (remove) {
        RemovePoints(inputCloud, inlierIndices);
    }
    Eigen::Vector4f floorParameters;
    float curve;
    pcl::computePointNormal(*inputCloud, inlierIndices, floorParameters, curve);
    return floorParameters;
}

std::vector<int> ObjectDetection::FindOrthogonalPlaneRansac (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud, Eigen::Vector3f normal, float epsAngle, float distThreshold) {
    // setup parallel plane model
    this->parallelPlaneModel->setInputCloud(inputCloud);
    this->parallelPlaneModel->setAxis(normal);
    this->parallelPlaneModel->setEpsAngle(pcl::deg2rad(epsAngle));
    // setup parallel plane ransac
    this->parallelPlaneRansac->setDistanceThreshold(distThreshold);
    // get first plane
    this->parallelPlaneRansac->computeModel();
    //DrawPlane(modelCoeffs, "cuboid side 1");
    std::vector<int> planeIndices;
    this->parallelPlaneRansac->getInliers(planeIndices);
    return planeIndices;
}

float ObjectDetection::MaxExtent (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud) {
    pcl::PointXYZ centroid;
    pcl::computeCentroid(*inputCloud, centroid);
    float maxExtent = -1.0;
    if (inputCloud->size()) {
        for (size_t i = 0; i != inputCloud->size(); ++i) {
            float eucDist = pcl::euclideanDistance(inputCloud->points[i], centroid);
            if (eucDist > maxExtent) {
                maxExtent = eucDist;
            }
        }
        RCLCPP_DEBUG(get_logger(), "max extent of cloud is %f", maxExtent);
    }
    return maxExtent;
}

int main (int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetection>());
    rclcpp::shutdown();
    return 0;
}
