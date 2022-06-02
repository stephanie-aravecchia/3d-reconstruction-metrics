#include "comparator/compute_ot_metrics.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <ros/ros.h>
#define DEBUG false

using namespace std;

ComputeOTMetrics::ComputeOTMetrics(unsigned int i, double reg, int num, double stop): 
        cube_nrows_(1), cube_ncols_(i) , cube_nslices_(1), 
        reg_(reg), numItermax_(num), stopThres_(stop) {
    unsigned int dim = cube_nrows_*cube_ncols_*cube_nslices_;
    M_.resize(dim, dim);
    uniform_ = Eigen::VectorXd::Constant(dim, 1.)/dim;
}
ComputeOTMetrics::ComputeOTMetrics(unsigned int i, unsigned int j, double reg, int num, double stop): 
        cube_nrows_(i), cube_ncols_(j) , cube_nslices_(1),
        reg_(reg), numItermax_(num), stopThres_(stop) {
    unsigned int dim = cube_nrows_*cube_ncols_*cube_nslices_;
    M_.resize(dim, dim);
    uniform_ = Eigen::VectorXd::Constant(dim, 1.)/dim;
}
ComputeOTMetrics::ComputeOTMetrics(unsigned int i, unsigned int j, unsigned int k, double reg, int num, double stop): 
        cube_nrows_(j), cube_ncols_(k) , cube_nslices_(i),
        reg_(reg), numItermax_(num), stopThres_(stop) {
    unsigned int dim = cube_nrows_*cube_ncols_*cube_nslices_;
    M_.resize(dim, dim);
    uniform_ = Eigen::VectorXd::Constant(dim, 1.)/dim;
}

void ComputeOTMetrics::setCostMatrixSquareDist() {
    for (unsigned int k{0}; k <cube_nslices_; ++k) {
        for (unsigned int j{0}; j <cube_nrows_; ++j) {
            for (unsigned int i{0}; i <cube_ncols_; ++i) {
                unsigned int row_index = k*(cube_ncols_*cube_nrows_) + j*cube_nrows_ + i;
                for (unsigned int z{0}; z <cube_nslices_; ++z) {
                    for (unsigned int y{0}; y <cube_nrows_; ++y) {
                        for (unsigned int x{0}; x <cube_ncols_; ++x) {
                            unsigned int col_index = z*(cube_ncols_*cube_nrows_) + y*cube_nrows_ + x;
                            M_(row_index, col_index) = (x-i)*(x-i) + (y-j)*(y-j) + (z-k)*(z-k);
                        }     
                    }
                }
            }
        } 
    }
    //we also set K_
    K_ = Eigen::exp(M_.array()/(-reg_));
    std::cout << "cost Matrix shape is rows, cols: " << M_.rows() << ", " << M_.cols() << std::endl;
}

void ComputeOTMetrics::normalizeAndAddNoise(Eigen::VectorXd& v) const {
    v = v.array() + 1e-6;
    assert(fabs(v.sum())>=1e-6);
    v = v.array()/v.sum();
}
void ComputeOTMetrics::normalizeAndAddNoise(const Eigen::VectorXd& v,Eigen::VectorXd& nv) const {
    Eigen::ArrayXd v_arr = v.array() + 1e-6;
    assert(fabs(v_arr.sum())>=1e-6);
    v_arr/= v_arr.sum();
    nv = v_arr.matrix();
}


void ComputeOTMetrics::getOccAndFreeDistri(const Eigen::VectorXd& input, Eigen::VectorXd& occ, Eigen::VectorXd& free) const {
    assert(input.size()>0);
    Eigen::VectorXd mapped = 2 * input.array() -1;
    occ = Eigen::VectorXd::Constant(input.size(), 0);
    free = Eigen::VectorXd::Constant(input.size(), 0);
    for (int i{0}; i <input.size(); ++i) {
        if (mapped[i] > 0) {
            occ[i] = mapped[i];
        } else if (mapped[i] < 0) {
            free[i] = mapped[i];
        } else {
            continue;
        }
    }
    free = -1 * free.array();
}

ComputeOTMetrics::OccFreeWasserstein ComputeOTMetrics::normalizeAndGetSinkhornDistanceSigned(std::vector<double>& a, std::vector<double>& b) const {
    assert(a.size() == b.size());
    double* ptr= &(a.front());
    size_t size = a.size();
    Eigen::Map<Eigen::VectorXd> a_eigen(ptr, size);
    double* ptr1= &(b.front());
    Eigen::Map<Eigen::VectorXd> b_eigen(ptr1, size);
    
    Eigen::VectorXd a_occ;
    Eigen::VectorXd a_free;
    getOccAndFreeDistri(a_eigen, a_occ, a_free);
    normalizeAndAddNoise(a_occ); 
    normalizeAndAddNoise(a_free); 

    Eigen::VectorXd b_occ;
    Eigen::VectorXd b_free;
    getOccAndFreeDistri(b_eigen, b_occ, b_free);
    normalizeAndAddNoise(b_occ); 
    normalizeAndAddNoise(b_free); 

    ComputeOTMetrics::OccFreeWasserstein wd;
    wd.free = getSinkhornDistance(a_free, b_free);
    wd.occ = getSinkhornDistance(a_occ, b_occ);
    return wd;
}
double ComputeOTMetrics::normalizeAndGetSinkhornDistance(std::vector<double>& a, std::vector<double>& b) const {

    double* ptr= &(a.front());
    size_t size = a.size();
    Eigen::Map<Eigen::VectorXd> a_eigen(ptr, size);
    double* ptr1= &(b.front());
    size_t size1 = b.size();
    Eigen::Map<Eigen::VectorXd> b_eigen(ptr1, size1);
    return normalizeAndGetSinkhornDistance(a_eigen, b_eigen);
}

double ComputeOTMetrics::normalizeAndGetSinkhornDistance(const Eigen::VectorXd& a0, const Eigen::VectorXd& b0) const {
    //The sinkhorn algorithm is design to work on probabilities distribution (i.e with sum = 1).
    //Also pot implementation doesn't deal wiht zeros, so we add "noise" (actually a small constant)
    Eigen::VectorXd a;
    Eigen::VectorXd b; 
    normalizeAndAddNoise(a0, a);
    normalizeAndAddNoise(b0, b);
    return getSinkhornDistance(a, b);
}
//Implementation as in POT
double ComputeOTMetrics::getSinkhornDistance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const {
    //we don't deal with empty a or b
    //we also don't deal with multiple histograms in b
    assert(a.size()>0);
    assert(b.size()>0);
#if DEBUG
    std::cout << "a0, b0: " << a0 << "," << b0 << std::endl;
#endif
    int dim_a = a.size();
    int dim_b = b.size();

    Eigen::VectorXd u = Eigen::VectorXd::Constant(dim_a, 1.)/dim_a;
    Eigen::VectorXd v = Eigen::VectorXd::Constant(dim_b, 1.)/dim_b;

    Eigen::MatrixXd Kp = K_.array().colwise() * a.cwiseInverse().array();

    int cpt = 0;
    double err = 1;

#if DEBUG
    std::cout << "a, b: " << a << ", " << b << std::endl;
    std::cout << "M: " << M_ << std::endl;
    std::cout << "u, v: " << u << ", " << v << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "Kp: " << Kp << std::endl;
#endif

    while ((err>stopThres_) && (cpt < numItermax_)) {
        Eigen::VectorXd uprev = u;
        Eigen::VectorXd vprev = v;
        Eigen::MatrixXd KtransposeU = K_.transpose() *u;
#if DEBUG
        std::cout << "KtransposeU.rows: " << KtransposeU.rows() << std::endl;
        std::cout << "KtransposeU.cols: " << KtransposeU.cols() << std::endl;
        std::cout << "b.rows: " << b.rows() << std::endl;
        std::cout << "b.cols: " << b.cols() << std::endl;
#endif
        v = (b.array() / KtransposeU.array());
        u = (Kp * v).cwiseInverse();
#if DEBUG
        std::cout << "KtransposeU: " << KtransposeU << std::endl;
        std::cout << "u, v: " << u << ", " << v << std::endl;
#endif
        if ((((KtransposeU.array() == 0).any()) || (!(u.allFinite()))) || (!(v.allFinite()))) {
            std::cout << "Warning: numerical errors at iteration " <<  cpt << std::endl;
            u = uprev;
            v = vprev;
            break;
        }
        if ((cpt % 10) == 0) {
            Eigen::VectorXd tmp2 = ((K_.array().colwise() *u.array()).rowwise() *v.transpose().array()).colwise().sum().transpose();
            err = (tmp2 - b).norm();
#if DEBUG
            std::cout << "tmp2: " << tmp2 << std::endl;
            std::cout << "err: " << err << std::endl;
#endif
        }
        cpt++;
    }
    return (((K_.array().colwise() *u.array()).rowwise() *v.transpose().array()).array()*M_.array()).sum();
}
