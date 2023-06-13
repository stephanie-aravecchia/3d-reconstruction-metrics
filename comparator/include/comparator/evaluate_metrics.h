#ifndef EVALUATE_METRICS_H
#define EVALUATE_METRICS_H
#include "comparator/datatypes.h"
#include "comparator/compute_metrics.h"
#include "comparator/compute_ot_metrics.h"
#include "comparator/box_tools.h"
#include <iostream>
#include <assert.h>
#include <thread>
#include <algorithm>
#include <signal.h>
#include <unistd.h>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <array>
#include <fstream>
#include <thread>
#include <mutex>


/* 
* This class implements the evaluation of the different considered metrics on a cuboid
* Loads a single GT cuboid, initialize a Reconstruction from the GT
* Incrementally adds noise in the Reconstruction 
* - iterate randomly on the 1000 elements
* - apply one of the 4 noise models
* - after the 1000 iterations, the rec cuboid is:
    * NoiseModel::RANDOM_OCC all occupied
    * NoiseModel::RANDOM_FREE all free
    * NoiseModel::RANDOM_UNKNOWN all unknown
    * NoiseModel::RANDOM all random
    * NoiseModel::DIFFUSION the occupied vox are diffused (switch randomly two neighbors)
    * NoiseModel::FLIP free and occupied space are inverted
* - we compute the metrics at each iteration 
*/

#define N_NOISE_MODEL 7
#define N_GAUSSIAN_BLUR 4

class EvaluateMetricsOnCuboid {
    
    public:
        struct CuboidInfo {
            std::string cuboid_img_folder;
            std::string xp;
            std::string cuboid_output_folder;
            int id=-1;
            double res;
            CuboidInfo() {}
            CuboidInfo(const CuboidInfo& c):
                cuboid_img_folder(c.cuboid_img_folder), xp(c.xp), 
                cuboid_output_folder(c.cuboid_output_folder),
                id(c.id), res(c.res) {}
        };
        
        struct CuboidResults {
            std::string xp;
            int id=-1;
            std::array<ComparatorDatatypes::Metrics, N_NOISE_MODEL+N_GAUSSIAN_BLUR-1> results{};
            CuboidResults() {}
            CuboidResults(const std::string& xp, int id) : xp(xp), id(id) {}
        };

        //Datatype to store the results of the transformations on the 1000 iterations
        typedef std::array <CuboidResults, 1000> CuboidAllResults;
        
        std::array<ComparatorDatatypes::CovThresholds,4> cov_thresholds = std::array{
            ComparatorDatatypes::CovThresholds(0.8, 0.05),
            ComparatorDatatypes::CovThresholds(0.8, 0.1),
            ComparatorDatatypes::CovThresholds(0.7, 0.1),
            ComparatorDatatypes::CovThresholds(0.7, 0.15),
        };

    protected:
        //Current cuboid info:
        CuboidInfo cuboid_info_;
        //All the results for the single cuboid 
        CuboidAllResults cuboid_all_results_;
        std::string input_folder_;
        int cuboid_id_;
        
        double result_name_;

        //The BoxTool box, to manipulate the cuboid
        int box_size_ = 10;
        const int mat_sz_[3] = {box_size_,box_size_,box_size_};
        BoxTools box_ = BoxTools(box_size_,box_size_,box_size_);
        //The actual gt 3d mat
        cv::Mat_<uint8_t> cuboidGtMat_ = cv::Mat_<uint8_t>(3, mat_sz_);
        int n_gt_points_;
        double occupancy_thres_=0.8;
        double divergence_to_unknown_thres_ = 0.1;
        int img_width_ = box_size_;
        int img_height_ = box_size_;
        double img_res_;
        double non_observed_vi_= 0;
        double non_observed_cov_ = 0;
        double non_observed_acc_ = 0;
        double non_observed_l1_ = 500;
        double non_observed_dkl_ = 13200;
        double non_observed_wd_= 200;

        
        ComputeOTMetrics ot_metrics_;
        enum class NoiseModel{RANDOM_OCC, RANDOM_FREE, RANDOM_UNKNOWN, RANDOM_P, RANDOM_DIFFUSION, RANDOM_FLIP, 
                    GAUSSIAN};

        int gaussian_index_=0;        
        //We define the noise levels for GAUSSIAN NoiseModel:
        //the values are (sigma, ksize, additional noise)
        std::array<ComparatorDatatypes::NoiseOnGtParams, N_GAUSSIAN_BLUR> noise_on_gt_ = std::array{
            ComparatorDatatypes::NoiseOnGtParams(0.1,7,0),
            ComparatorDatatypes::NoiseOnGtParams(0.1,11,0),
            ComparatorDatatypes::NoiseOnGtParams(0.2,11,0.0),
            ComparatorDatatypes::NoiseOnGtParams(0.3,11,0.0)
        };

        ComparatorDatatypes::Metrics calcMetrics(const cv::Mat_<uint8_t>& gtMat, const cv::Mat_<uint8_t>& recMat) const;

        cv::Point3i getPointFromIndex(int index) const;
        void addNoise(int index, NoiseModel nm, cv::Mat_<uint8_t>& recMat) const;
        cv::Point3i getRandomNeighbour(const cv::Point3i& point, cv::Mat_<uint8_t>& recMat) const;
        void evaluateCubOnNoiseModel(const std::vector<int>&, int);
        void saveToDiskSelectedNoiseModel(int) const;

    public:
        EvaluateMetricsOnCuboid(const CuboidInfo& cuboid_info);
        void evaluate();
        void saveToDisk() const;
        const CuboidAllResults& getResults() const {
            return cuboid_all_results_;
        };



};

#endif