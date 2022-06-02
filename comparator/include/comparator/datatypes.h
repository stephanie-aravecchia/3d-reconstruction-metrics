#ifndef SLICE_OCTOMAP_DATATYPE_H
#define SLICE_OCTOMAP_DATATYPE_H
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

//Datatypes shared by the different classes in that package
class ComparatorDatatypes {
    public:
        
        //Metric bbox, values are double        
        struct BBox {
            double xmin;
            double xmax;
            double ymin;
            double ymax;
            double zmin;
            double zmax;
            BBox() {}
            BBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax):
                xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax) {}
        };
        
        //Pixel bbox, values are unsigned int        
        struct PixBBox {
            unsigned int xmin;
            unsigned int xmax;
            unsigned int ymin;
            unsigned int ymax;
            unsigned int zmin;
            unsigned int zmax;
            PixBBox() {}
            PixBBox(int xmin, int xmax, int ymin, int ymax, int zmin, int zmax):
                xmin(static_cast<unsigned int>(xmin)), xmax(static_cast<unsigned int>(xmax)), 
                ymin(static_cast<unsigned int>(ymin)), ymax(static_cast<unsigned int>(ymax)), 
                zmin(static_cast<unsigned int>(zmin)), zmax(static_cast<unsigned int>(zmax)) {}
            PixBBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax):
                xmin(static_cast<unsigned int>(xmin)), xmax(static_cast<unsigned int>(xmax)), 
                ymin(static_cast<unsigned int>(ymin)), ymax(static_cast<unsigned int>(ymax)), 
                zmin(static_cast<unsigned int>(zmin)), zmax(static_cast<unsigned int>(zmax)) {}
        };
        
        //Pixel Offsets, in int
        struct Offsets {
            int x;
            int y;
            int z;
            Offsets() {}
            Offsets(double x, double y, double z): x(std::round(x)), y(std::round(y)), z(std::round(z)) {}
            Offsets(float x, float y, float z): x(std::round(x)), y(std::round(y)), z(std::round(z)) {}
            Offsets(int x, int y, int z): x(x), y(y), z(z) {}
            Offsets(int x, int y): x(x), y(y), z(0) {}
        };
        
        //Reconstruction metrics to compare two 3d-grids
        struct Metrics {
            double surface_coverage;
            double reconstruction_accuracy;
            double volumetric_information;
            double emptyness_entropy;
            double emptyness_l1;
            double emptyness_l2;
            double dkl;
            double wasserstein_occ_remapped;
            double wasserstein_free_remapped;
            double wasserstein_direct;
            int n_match;;
            int n_rec_points;
            int n_gt_points;
            Metrics():surface_coverage(-10), reconstruction_accuracy(-10), volumetric_information(-10), emptyness_entropy(-10), emptyness_l1(-10), emptyness_l2(-10), dkl(-10),
                        wasserstein_occ_remapped(-10), wasserstein_free_remapped(-10), wasserstein_direct(-10),
                        n_match(-10), n_rec_points(-10), n_gt_points(-10) {}
            Metrics(int v) {
                if (v <0) {
                    surface_coverage = -1;
                    reconstruction_accuracy = -1;
                    volumetric_information = -1;
                    emptyness_entropy = -1;
                    emptyness_l1 = -1;
                    emptyness_l2 = -1;
                    dkl = -1;
                    wasserstein_occ_remapped = -1;
                    wasserstein_free_remapped = -1;
                    wasserstein_direct = -1;
                    n_match = -1;
                    n_rec_points = -1;
                    n_gt_points = -1;
                } else {
                    std::cout << "Error in Metrics constructor, assumed called with a negative value" << std::endl;
                    surface_coverage = -1;
                    reconstruction_accuracy = -1;
                    volumetric_information = -1;
                    emptyness_entropy = -1;
                    emptyness_l1 = -1;
                    emptyness_l2 = -1;
                    dkl = -1;
                    wasserstein_occ_remapped = -1;
                    wasserstein_free_remapped = -1;
                    wasserstein_direct = -1;
                    n_match = -1;
                    n_rec_points = -1;
                    n_gt_points = -1;
                } 
            }
        };

        //Reconstruction metrics to compare two 3d-grids
        //Old version still used in Limit
        struct MetricsOld {
            double min;
            double max;
            double sum;
            double hausdorff;
            double dkl;
            double wasserstein_occ;
            double wasserstein_free;
            double gt_occupancy_rate;
            int n_obs;
            MetricsOld():min(-10), max(-10),sum(-10),hausdorff(-10),dkl(-10), wasserstein_occ(-10), wasserstein_free(-10), gt_occupancy_rate(-10), n_obs(-10) {}
            MetricsOld(int v) {
                if (v <0) {
                    min = -1;
                    max = -1;
                    sum = -1;
                    hausdorff = -1;
                    dkl = -1;
                    wasserstein_occ = -1;
                    wasserstein_free = -1;
                    gt_occupancy_rate = -1;
                    n_obs = 0;
                } else {
                    std::cout << "Error in Metrics constructor, assumed called with a negative value" << std::endl;
                    min = -1;
                    max = -1;
                    sum = -1;
                    hausdorff = -1;
                    dkl = -1;
                    wasserstein_occ = -1;
                    wasserstein_free = -1;
                    gt_occupancy_rate = -1;
                    n_obs = 0;
                } 
            }
            MetricsOld(double min, double max, double sum, double hausdorff, double dkl, double wasserstein_occ, double wasserstein_free, double gt_occupancy_rate, unsigned int n_obs) :
                min(min), max(max), sum(sum), hausdorff(hausdorff), dkl(dkl), wasserstein_occ(wasserstein_occ), wasserstein_free(wasserstein_free), gt_occupancy_rate(gt_occupancy_rate), n_obs(n_obs) {}
        };

        struct BoxToProcess {
            int index_in_vect;
            std::vector<cv::Range> ranges;
            Metrics metrics;
            BBox mbbox;
            BoxToProcess() {}
        };
        struct BoxToProcessLimitOnly {
            int index_in_vect;
            std::vector<cv::Range> ranges;
            MetricsOld metrics_ideal;
            MetricsOld metrics_random;
            MetricsOld metrics_unknown;
            BBox mbbox;
            BoxToProcessLimitOnly() {}
        };

    friend std::ostream& operator<<(std::ostream& out, const BBox& bbox) {
        out << "xmin: " << bbox.xmin << " " 
            << "xmax: " << bbox.xmax << " " 
            << "ymin: " << bbox.ymin << " " 
            << "ymax: " << bbox.ymax << " " 
            << "zmin: " << bbox.zmin << " " 
            << "zmax: " << bbox.zmax;
        return out;
    } 
    friend std::ostream& operator<<(std::ostream& out, const PixBBox& bbox) {
        out << "xmin: " << bbox.xmin << " " 
            << "xmax: " << bbox.xmax << " " 
            << "ymin: " << bbox.ymin << " " 
            << "ymax: " << bbox.ymax << " " 
            << "zmin: " << bbox.zmin << " " 
            << "zmax: " << bbox.zmax;
        return out;
    } 
    friend std::ostream& operator<<(std::ostream& out, const Offsets& offset) {
        out << "x: " << offset.x << " " 
            << "y: " << offset.y << " " 
            << "z: " << offset.z;
        return out;
    } 
    
};

//We also overload the << operator to print vectors
std::ostream& operator<<(std::ostream& out, const std::vector<double>& vect) {
    if (vect.size()>1) {
        for (auto i = vect.cbegin(); i != (vect.cend()-1); ++i) {
            out << *i << ",";
        }
        out << *(vect.cend()-1);
    } else {
        out << *(vect.cbegin());
    }
    return out;
} 


#endif