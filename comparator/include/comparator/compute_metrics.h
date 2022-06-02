#ifndef COMPUTE_METRICS_H
#define COMPUTE_METRICS_H
#include <vector>
#include <Eigen/Eigen>
#include "comparator/datatypes.h"


//Here we implement some metrics used in compare rec gt
namespace ComputeMetrics {
    
        void normalize(const std::vector<double>&, std::vector<double>&);
        double getKLDivergence(const std::vector<double>&, const std::vector<double>&); 
        double getKLDivergenceOnDistri(const std::vector<double>&, const std::vector<double>&); 
        
        //The volumetric information within a voxel can be defined as its entropy (Scaramuzza paper)
        //The entropy of the cube is the sum
        double getVolumetricInformation(const std::vector<double>&); 
        //Calculate the KL divergence between an empty gt and the reconstruction
        double getEmptynessEntropy(const std::vector<double>&); 
        //other measure of the emptyness reconstruction metrics
        double getEmptynessL1(const std::vector<double>&);
        double getEmptynessL2(const std::vector<double>&);
        //debug function to make sure GT is composed only of 0 and 1
        bool isGTVectorBinary(const std::vector<double>&);
        bool isGTImageBinary(const cv::Mat_<uint8_t>&);
        bool isValue0or1(double);
        bool isValue0or1(uint8_t);
        //The surface coverage was computed by matching the reconstructed point cloud to that of the original model [31].
        //If the distance between an original point and its closest reconstructed point was less than the registration distance of 0.05m,
        //the original point was considered as observed
        //The surface coverage cs is then the percentage of observed surface points compared to the total number of surface points of the model:
        //(from above, with different registration distance and occupancy thresholds)
        double getSurfaceCoverage(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& gtMat, const cv::Mat_<uint8_t>& recMat, 
                        double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res);
        double getReconstructionAccuracy(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& gtMat, const cv::Mat_<uint8_t>& recMat, 
                        double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res);
        double getBoxDistMetrics(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& refMat, const cv::Mat_<uint8_t>& queryMat, 
                        double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res);
        int getBoxNMatch(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& refMat, const cv::Mat_<uint8_t>& queryMat, 
                        double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res);
        //returns the number of points in the box above the occupancy threshold
        int getNpoints(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& refMat, double occ_thres, 
                        int img_width, int img_height, int box_size);
        int getNpoints(const std::vector<double>& vect, double occ_thres);
        
        //utility function to normalize double to color and the inverse
        uint8_t getNormalizedPixValue(double v);
        double pix2proba(uint8_t);
};

#endif