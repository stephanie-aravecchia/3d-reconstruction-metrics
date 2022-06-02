#include "comparator/compute_metrics.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <numeric>

using namespace std;

double ComputeMetrics::getKLDivergence(const std::vector<double>& rec, const std::vector<double>& gt) {
    double dkl_sum{0};
    assert(rec.size() == gt.size());
    for (size_t i{0}; i < rec.size(); i++) {
        double dkl{0};
        //if Gt is 1:
        if (fabs(gt[i])>(1-1e-6)) {
            dkl = (1-rec[i])*log((1-rec[i])/1e-6) + rec[i]*log(rec[i]/(1-1e-6));
        //if Gt is 0:
        } else if (fabs(gt[i])<(1e-6)) {
            dkl = (1-rec[i])*log((1-rec[i])/(1-1e-6)) + rec[i]*log(rec[i]/1e-6);
        } else {
            std::cerr << "gt is not binary: " << gt[i] << std::endl;
            //add an assert so the code fails
            //gt = 1 -> fabs(gt-1) <=1e-6)
            //gt = 0 -> fabs(gt) <=1e-6)
            assert((fabs(gt[i]-1) <=1e-6) || (fabs(gt[i])<=1e-6));
        }
        dkl_sum += dkl;
    }
    return dkl_sum;
}

bool ComputeMetrics::isValue0or1(uint8_t v){
    if (v==0) {
        return true;
    } else if (v==255) {
        return true;
    }
    return false; 
    //if ((v!=0) && (v!=255)) {
    //    return false;
    //}
    //return true;
}

bool ComputeMetrics::isValue0or1(double v){
    if (fabs(v-1) < 1e-6) {
        return true;
    } else if (fabs(v)<1e-6) {
        return true;
    }
    return false;
}

bool ComputeMetrics::isGTImageBinary(const cv::Mat_<uint8_t>& mat) {
    for (int i{0}; i < mat.size().width; ++i) {
        for (int j{0}; j < mat.size().height; ++j) {
            if (!isValue0or1(mat.at<uint8_t>(j,i))) {
                std::cerr << "gt image is not binary:  value is: "<< mat.at<uint8_t>(j,i) << ". i: "<< i << ",j: "<< j << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool ComputeMetrics::isGTVectorBinary(const std::vector<double>& gt) {
    for (size_t i{0}; i < gt.size(); i++) {
        if (isValue0or1(gt[i])) {
            continue;
        } else {
            std::cerr << "gt vector is not binary: " << gt[i] << std::endl;
            return false;
        }
    }
    return true;
}

double ComputeMetrics::getKLDivergenceOnDistri(const std::vector<double>& rec, const std::vector<double>& gt) {
    double dkl_sum{0};
    assert(rec.size() == gt.size());
    for (size_t i{0}; i < rec.size(); i++) {
        double dkl{0};
        //if Gt is 1:
        if (fabs(gt[i])>(1-1e-6)) {
            dkl = (1-rec[i])*log((1-rec[i])/1e-6) + rec[i]*log(rec[i]/(1-1e-6));
        //if Gt is 0:
        } else if (fabs(gt[i])<(1e-6)) {
            dkl = (1-rec[i])*log((1-rec[i])/(1-1e-6)) + rec[i]*log(rec[i]/1e-6);
        } else {
            dkl = (1-rec[i])*log((1-rec[i])/(1-gt[i])) + rec[i]*log(rec[i]/gt[i]);
        }
        dkl_sum += dkl;
    }
    return dkl_sum;
}

void ComputeMetrics::normalize(const std::vector<double>& vect, std::vector<double>& norm) {
    double sum = std::accumulate(vect.begin(), vect.end(), 0.0);
    if (sum == 0) {
        norm = std::vector<double>(vect.size(), 1./vect.size());
    } else {
        norm.clear();
        for (const double& p : vect) {
            norm.push_back(p/sum);
        }
    }
}

double ComputeMetrics::getVolumetricInformation(const std::vector<double>& vect) {
    //we calculate the entropy of each element of the voxel, and we sum it on the vector
    //the entropy of an element is symetric around .5
    double vi{0};
    for (double p : vect) {
        if (p<1e-6){
            p = 1e-6;
        }else if (p>(1-1e-6)) {
            p = 1-1e-6;
        }
        vi += -p*log(p)-(1-p)*log(1-p);
    }
    return vi;
}
double ComputeMetrics::getEmptynessEntropy(const std::vector<double>& vect) {
    //This is actually the KL divergence between the reconstruction and an empy ground-truth
    //KL divergence is for probability distribution, we normalize
    std::vector<double> gt0(vect.size(),1./vect.size());
    std::vector<double> vect_normalized;
    normalize(vect, vect_normalized);
    return getKLDivergenceOnDistri(vect_normalized, gt0);
}
double ComputeMetrics::getEmptynessL1(const std::vector<double>& vect) {
    //This is the sum of the probabilities in the vector
    double tot{0};
    for (const double& p : vect) {
        tot += p;
    }
    return tot;
}
double ComputeMetrics::getEmptynessL2(const std::vector<double>& vect) {
    //This is the square root of the sum of the squared probabilities in the vector
    double tot{0};
    for (const double& p : vect) {
        tot += (p*p);
    }
    return sqrt(tot);
}


double ComputeMetrics::getReconstructionAccuracy(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& gtMat, const cv::Mat_<uint8_t>& recMat, 
                double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res) {
    //we loop on all the points from the reconstruction
    //we consider only points with an occupancy > occ_thres
    //for these points, if the distance to the closest ground_truth is smaller than reg_dist
        //we consider the point reconstructed accurately
        //we return the ratio number of accurate reconstructions / total of reconstructions
        return getBoxDistMetrics(box, recMat, gtMat, reg_dist, occ_thres, img_width, img_height, box_size, img_res);
}
double ComputeMetrics::getSurfaceCoverage(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& gtMat, const cv::Mat_<uint8_t>& recMat, 
                double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res) {
    //we loop on all the points from the ground_truth
    //we consider only occupied point in the ground truth
    //for these points, if the distance to the closest reconstructed (>occ_thres) is smaller than reg_dist
        //we consider the point reconstructed
        //we return the ratio number of reconstructions / total of gt points
        return getBoxDistMetrics(box, gtMat, recMat, reg_dist, occ_thres, img_width, img_height, box_size, img_res);
}
int ComputeMetrics::getBoxNMatch(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& refMat, const cv::Mat_<uint8_t>& queryMat, 
                double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res) {
    //we loop on all the points from the query
    //we consider only points with an occupancy > occ_thres
    //for these points, if the distance to the closest refMat is smaller than reg_dist
        //we consider the points matching the above test
        //we return the ratio number of match / total of points in ref
    int n_match{0};
    int n_points{0};
    uint8_t pixthres{getNormalizedPixValue(occ_thres)};
    for (int i{box.ranges[0].start}; i<box.ranges[0].end; i++) {
        for (int j{box.ranges[1].start}; j<box.ranges[1].end; j++) {
            for (int k{box.ranges[2].start}; k<box.ranges[2].end; k++) {
                assert(i>=0);
                assert(j>=0);
                assert(k>=0);
                assert(i<img_width);
                assert(j<img_height);
                assert(k<box_size);
                if (queryMat(i,j,k) > pixthres) {
                    ++n_points;
                    //we look for the closest in ref:
                    bool is_match = false;
                    int x = box.ranges[0].start;
                    while ((x<box.ranges[0].end) && (!is_match)) {
                        int y = box.ranges[1].start;
                        while ((y<box.ranges[1].end) && (!is_match)) {
                            int z = box.ranges[2].start;
                            while ((z<box.ranges[2].end) && (!is_match)) {
                                assert(x>=0);
                                assert(y>=0);
                                assert(z>=0);
                                assert(x<img_width);
                                assert(y<img_height);
                                assert(z<box_size);
                                if (refMat(x,y,z) > pixthres) {
                                    double dist = sqrt(static_cast<double>(((x-i)*(x-i)+(y-j)*(y-j)+(z-k)*(z-k))))*img_res;
                                    if (dist <= reg_dist) {
                                        ++n_match;
                                        is_match = true;
                                    }
                                }
                                ++z;
                            }
                            ++y;
                        }
                        ++x;
                    }
                }
            }
        }
    }
    if (n_points == 0) {
        return 0;
    } else {
        return n_match;
    }
}
double ComputeMetrics::getBoxDistMetrics(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& refMat, const cv::Mat_<uint8_t>& queryMat, 
                double reg_dist, double occ_thres, int img_width, int img_height, int box_size, double img_res) {
    //we loop on all the points from the query
    //we consider only points with an occupancy > occ_thres
    //for these points, if the distance to the closest refMat is smaller than reg_dist
        //we consider the points matching the above test
        //we return the ratio number of match / total of points in ref
    int n_match{0};
    int n_points{0};
    for (int i{box.ranges[0].start}; i<box.ranges[0].end; i++) {
        for (int j{box.ranges[1].start}; j<box.ranges[1].end; j++) {
            for (int k{box.ranges[2].start}; k<box.ranges[2].end; k++) {
                assert(i>=0);
                assert(j>=0);
                assert(k>=0);
                assert(i<img_width);
                assert(j<img_height);
                assert(k<box_size);
                if (queryMat(i,j,k) > getNormalizedPixValue(occ_thres)) {
                    ++n_points;
                    //we look for the closest in ref:
                    bool is_match = false;
                    int x = box.ranges[0].start;
                    while ((x<box.ranges[0].end) && (!is_match)) {
                        int y = box.ranges[1].start;
                        while ((y<box.ranges[1].end) && (!is_match)) {
                            int z = box.ranges[2].start;
                            while ((z<box.ranges[2].end) && (!is_match)) {
                                assert(x>=0);
                                assert(y>=0);
                                assert(z>=0);
                                assert(x<img_width);
                                assert(y<img_height);
                                assert(z<box_size);
                                if (refMat(x,y,z) > getNormalizedPixValue(occ_thres)) {
                                    double dist = sqrt(static_cast<double>(((x-i)*(x-i)+(y-j)*(y-j)+(z-k)*(z-k))))*img_res;
                                    if (dist <= reg_dist) {
                                        ++n_match;
                                        is_match = true;
                                    }
                                }
                                ++z;
                            }
                            ++y;
                        }
                        ++x;
                    }
                }
            }
        }
    }
    if (n_points == 0) {
        return 0;
    } else {
        return static_cast<double>(n_match)/n_points;
    }
}

int ComputeMetrics::getNpoints(const ComparatorDatatypes::BoxToProcess& box, const cv::Mat_<uint8_t>& refMat, double occ_thres, 
        int img_width, int img_height, int box_size){ 
    int n_points{0};
    for (int i{box.ranges[0].start}; i<box.ranges[0].end; i++) {
        for (int j{box.ranges[1].start}; j<box.ranges[1].end; j++) {
            for (int k{box.ranges[2].start}; k<box.ranges[2].end; k++) {
                assert(i>=0);
                assert(j>=0);
                assert(k>=0);
                assert(i<img_width);
                assert(j<img_height);
                assert(k<box_size);
                if (refMat(i,j,k) > getNormalizedPixValue(occ_thres)) {
                    ++n_points;
                }
            }
        }
    }
    return n_points;
}

int ComputeMetrics::getNpoints(const std::vector<double>& vect, double occ_thres) {
    int n_points{0};
    for (const double& p : vect) {
        if (p > occ_thres){
            ++n_points;
        }
    }
    return n_points;
}

uint8_t ComputeMetrics::getNormalizedPixValue(double v) {
    return static_cast<uint8_t>(std::max(0., std::min((v*255), 255.)));
}
double ComputeMetrics::pix2proba(uint8_t v) {
    return std::max(0., std::min(v/255.,1.0));
}
