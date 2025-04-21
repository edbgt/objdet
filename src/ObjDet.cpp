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
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_parallel_plane.h>
#include <pcl/sample_consensus/sac_model_perpendicular_plane.h>

#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "ObjDet.hpp"

ObjectDetection::ObjectDetection () : rclcpp::Node ("ObjectDetectionNode") {
    subscription = this->create_subscription<sensor_msgs::msg::PointCloud2>("tof_point_cloud", 1, std::bind(&ObjectDetection::PointCloudReceivedCallback, this, std::placeholders::_1));
    rclcpp::on_shutdown(std::bind(&ObjectDetection::Stop, this));
    this->Start();
}

void ObjectDetection::PointCloudReceivedCallback (const sensor_msgs::msg::PointCloud2 & msg) {
    RCLCPP_DEBUG(get_logger(), "received point cloud of size %u * %u", msg.width, msg.height);
    // convert msg to pc
    pcl::fromROSMsg(msg, *(this->cloud));
    // show in viewer
    this->viewer->removeAllPointClouds();
    this->viewer->removeAllShapes();

    if (this->frameCounter < 10) {
        // during first 10 frames determine floor normal
        this->floorParams = CalculateFloorNormal(this->cloud, true);
        this->frameCounter++;
        RCLCPP_INFO(get_logger(), "determining floor normal");
    } else if (this->frameCounter == 10) {
        DrawPlane(floorParams, "floor");
        this->floorNormal = {this->floorParams.x(), this->floorParams.y(), this->floorParams.z()};
        this->frameCounter++;
        RCLCPP_INFO(get_logger(), "floor normal determined");
    } else {
        DrawPlane(floorParams, "floor");
        // reduce number of points in cloud
        RemoveFarPoints(0.6); // m
        RemoveClosePoints(0.3); // m
        this->viewer->addPointCloud<pcl::PointXYZ>(this->cloud, "received cloud", 0);
        std::vector<int> firstPlaneIndices = FindOrthogonalPlane(floorNormal, 1.0, 0.01);
        pcl::RGB firstRgb (255, 0, 255), secondRgb (0, 255, 255);
        pcl::PointCloud<pcl::PointXYZ>::Ptr first (new pcl::PointCloud<pcl::PointXYZ> ());
        MaxExtent(first);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> firstColor (first, firstRgb.r, firstRgb.g, firstRgb.b);
        pcl::copyPointCloud(*(this->cloud), firstPlaneIndices, *first);
        if (!this->viewer->addPointCloud<pcl::PointXYZ> (first, firstColor, "first cuboid side")) {
            RCLCPP_ERROR(get_logger(), "could not add first cuboid side point cloud");
        }
        RemovePoints(firstPlaneIndices);
    }
    this->viewer->spinOnce();
}

void ObjectDetection::Start () {
    RCLCPP_INFO(get_logger(), "starting");
    this->frameCounter = 0;
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ> ());
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
        RemovePoints (inlierIndices);
    }
    Eigen::Vector4f floorParameters;
    float curve;
    pcl::computePointNormal(*inputCloud, inlierIndices, floorParameters, curve);
    return floorParameters;
}

std::vector<int> ObjectDetection::FindOrthogonalPlane (Eigen::Vector3f normal, float epsAngle, float distThreshold) {
    // setup parallel plane model
    this->parallelPlaneModel->setInputCloud(this->cloud);
    this->parallelPlaneModel->setAxis(normal);
    this->parallelPlaneModel->setEpsAngle(pcl::deg2rad(epsAngle));
    // setup parallel plane ransac
    this->parallelPlaneRansac->setDistanceThreshold(distThreshold);
    // get first plane
    this->parallelPlaneRansac->computeModel();
    Eigen::VectorXf modelCoeffs;
    this->parallelPlaneRansac->getModelCoefficients(modelCoeffs);
    RCLCPP_DEBUG(get_logger(), "model coefficients of plane: %f\t%f\t%f\t%f", modelCoeffs[0], modelCoeffs[1], modelCoeffs[2], modelCoeffs[3]);
    //DrawPlane(modelCoeffs, "cuboid side 1");
    std::vector<int> planeIndices;
    this->parallelPlaneRansac->getInliers(planeIndices);
    return planeIndices;
}

float ObjectDetection::MaxExtent (pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud) {
    pcl::PointXYZ centroid;
    pcl::computeCentroid(*inputCloud, centroid);
    float maxExtent = 0.0;
    RCLCPP_DEBUG(get_logger(), "size of cloud is %lu", inputCloud->size());
    for (size_t i = 0; i != inputCloud->size(); ++i) {
        float eucDist = pcl::euclideanDistance(inputCloud->points[i], centroid);
        RCLCPP_DEBUG(get_logger(), "computed euclidean distance %f", eucDist);
        if (eucDist > maxExtent) {
            RCLCPP_DEBUG(get_logger(), "new max extent is %f", maxExtent);
            maxExtent = eucDist;
        }
    }
    RCLCPP_DEBUG(get_logger(), "max extent of cloud is %f", maxExtent);
    return maxExtent;
}

void ObjectDetection::EstimateNormalsBoundaries () {
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::Boundary>::Ptr boundaries;
    //boundaries->resize(this->cloud->size());
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> estimation;
    /*estimation.setInputCloud(this->cloud);
    estimation.setInputNormals(normals);
    estimation.setRadiusSearch(0.005); // 5 mm
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree (new pcl::search::KdTree<pcl::PointXYZ>);
    estimation.setSearchMethod(kdTree);
    estimation.compute(*boundaries);
    for (size_t i = 0; i != this->cloud->size(); ++i) {
        if (boundaries->points[i].boundary_point != 0) {
            this->displayCloud->points[i].r = 255;
            this->displayCloud->points[i].g = 0;
            this->displayCloud->points[i].b = 0;
        } else {
            this->displayCloud->points[i].r = 0;
            this->displayCloud->points[i].g = 0;
            this->displayCloud->points[i].b = 0;
        }
    }*/
    //this->viewer->addPointCloud(displayCloud, "boundaries", 0);
}

int main (int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetection>());
    rclcpp::shutdown();
    return 0;
}
