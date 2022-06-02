#ifndef COMPUTE_OT_METRICS_H
#define COMPUTE_OT_METRICS_H
#include <vector>
#include <array>
#include <random>
#include <Eigen/Eigen>

//This class is implementing some optimal transport algorithm as in POT
//R. Flamary and al., “POT python optimal transport library,” Journal of Machine Learning Research, vol. 22(78), pp. 1–8, 2021
class ComputeOTMetrics {
    
    protected:
        unsigned int cube_nrows_; 
        unsigned int cube_ncols_; 
        unsigned int cube_nslices_; 
        Eigen::MatrixXd M_;
        Eigen::MatrixXd K_;
        Eigen::VectorXd uniform_;
        double reg_;
        int numItermax_;
        double stopThres_;
    
    public:
        ComputeOTMetrics() {};
        //1d constructor cols , reg, numItermax, stopThres
        ComputeOTMetrics(unsigned int, double, int, double);
        //2d constructor rows, cols, reg, numItermax, stopThres
        ComputeOTMetrics(unsigned int, unsigned int, double, int, double);
        //3d constructor slices, rows, cols, reg, numItermax, stopThres
        ComputeOTMetrics(unsigned int, unsigned int, unsigned int, double, int, double);

        struct OccFreeWasserstein {
            double occ;
            double free;
            OccFreeWasserstein() {}
            OccFreeWasserstein(double occ, double free):
                occ(occ), free(free) {}
        };

        //takes std::vectors as input, transform to Eigen::Vector, remap to [-1,1], split to positive and negative
        //distribution, normalize (add noise and sum to one) each distribution
        //and then perform the optimization on positive (occ) and negative (free), returns the wasserstein distance
        //on occupied spacea and on free space
        OccFreeWasserstein normalizeAndGetSinkhornDistanceSigned(std::vector<double>& a, std::vector<double>& b)const;
        
        //takes std::vectors as input, transform to Eigen::Vector and call the overloaded function
        double normalizeAndGetSinkhornDistance(std::vector<double>& a, std::vector<double>& b) const;
        
        //normalize the data (add noise and sum to one) and call getSinkhornDistance
        double normalizeAndGetSinkhornDistance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;
        
        //Sinkhorn algorithm, as in pot librairy
        double getSinkhornDistance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) const;
        
        void getOccAndFreeDistri(const Eigen::VectorXd&, Eigen::VectorXd&, Eigen::VectorXd&) const;

        //add noise and sum to one, when the input vector is a const, the 2nd passed by ref is normalized
        void normalizeAndAddNoise(const Eigen::VectorXd&, Eigen::VectorXd&) const;
        //add noise and sum to one
        void normalizeAndAddNoise(Eigen::VectorXd&) const;
        void setCostMatrixSquareDist();
};

#endif



