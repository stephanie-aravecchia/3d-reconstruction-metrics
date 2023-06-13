#include "comparator/evaluate_metrics.h"
#include <list>
#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
    #include <experimental/filesystem> 
    namespace fs = std::experimental::filesystem;
#else
      error "Missing the <filesystem> header."
#endif
using namespace std;


//This code is the implementation and the driver of Evaluate Metrics
//Loads a list of cuboids to process on the disk (expected as follows: xpname cuboid_id img_res)
    //Load the cuboids, evaluate the metrics, save the results to disk 

EvaluateMetricsOnCuboid::EvaluateMetricsOnCuboid(const CuboidInfo& cuboid_info):
    cuboid_info_(cuboid_info), img_res_(cuboid_info.res)
        {
            //Prepare the image list to load the cuboid:
            std::vector<std::string> imglist;
            for (int i=0; i< box_size_; i++) {
                char fname[40];
                sprintf(fname, "cuboid_gt_%06d__%02d.png", cuboid_info_.id, i);
                imglist.push_back(cuboid_info_.cuboid_img_folder + "/" + fname);
            }

            //Load the cuboid 
            box_.load3DMat(cuboidGtMat_, imglist);
            
            double ot_reg=1;
            double ot_stopthres=1e-9;
            int ot_maxiter=1000;

            //Initialize the ot metrics 
            ot_metrics_ = ComputeOTMetrics(box_size_, box_size_, box_size_, ot_reg, ot_maxiter, ot_stopthres);
            ot_metrics_.setCostMatrixSquareDist();

            //Initialize the results to default values
            for (auto& c : cuboid_all_results_) {
                c = CuboidResults(cuboid_info_.xp, cuboid_info_.id);
            }
        }

cv::Point3i EvaluateMetricsOnCuboid::getPointFromIndex(int index) const {
    int k = index / (box_size_*box_size_);
    int rem = index % (box_size_*box_size_);
    int j = rem / (box_size_);
    int i = rem % (box_size_);
    return cv::Point3i(i,j,k);
}

cv::Point3i EvaluateMetricsOnCuboid::getRandomNeighbour(const cv::Point3i& point, cv::Mat_<uint8_t>& recMat) const {
    cv::Point3i nb;
    static std::random_device rd;
    static std::default_random_engine engine(rd());
    static std::uniform_int_distribution<int> dist(-1,1);
    int i = dist(engine);
    int j = dist(engine);
    int k = dist(engine);
    nb.x = max(0, min(point.x+i, box_size_-1));
    nb.y = max(0, min(point.y+j, box_size_-1));
    nb.z = max(0, min(point.z+k, box_size_-1));
    assert(nb.x>=0);
    assert(nb.y>=0);
    assert(nb.z>=0);
    assert(nb.x<box_size_);
    assert(nb.y<box_size_);
    assert(nb.z<box_size_);
    return nb;
}

void EvaluateMetricsOnCuboid::addNoise(int index, NoiseModel nm, cv::Mat_<uint8_t>& recMat) const {
    static std::random_device rd;
    static std::default_random_engine engine(rd());
    static std::uniform_int_distribution<int> dist(0,255);
    uint8_t p;
    cv::Point3i point = getPointFromIndex(index);
    assert(point.x>=0);
    assert(point.y>=0);
    assert(point.z>=0);
    assert(point.x<box_size_);
    assert(point.y<box_size_);
    assert(point.z<box_size_);
    switch (nm) {
    case NoiseModel::RANDOM_OCC:
        p = 255;
        recMat(point.x, point.y, point.z) = p;
        break;
    case NoiseModel::RANDOM_FREE:
        p = 0;
        recMat(point.x, point.y, point.z) = p;
        break;
    case NoiseModel::RANDOM_UNKNOWN:
        p = 127;
        recMat(point.x, point.y, point.z) = p;
        break;
    case NoiseModel::RANDOM_P:
        p = static_cast<uint8_t>(dist(engine));
        recMat(point.x, point.y, point.z) = p;
        break;
    case NoiseModel::RANDOM_DIFFUSION:
        {
        cv::Point3i nb = getRandomNeighbour(point, recMat);
        while(nb==point) {
            nb = getRandomNeighbour(point, recMat);
        }
        p = recMat(nb.x, nb.y, nb.z);
        uint8_t q = recMat(point.x, point.y, point.z);
        recMat(point.x, point.y, point.z) = p;
        recMat(nb.x, nb.y, nb.z) = q;
        }
        break;
    case NoiseModel::RANDOM_FLIP:
        p = recMat(point.x, point.y, point.z); 
        recMat(point.x, point.y, point.z) = static_cast<uint8_t>(255 - p);
        break;
    case NoiseModel::GAUSSIAN:
        {
        double sigma = noise_on_gt_.at(gaussian_index_).sigma;
        double ksize = noise_on_gt_.at(gaussian_index_).ksize;
        double unif = noise_on_gt_.at(gaussian_index_).additionnal_uniform_noise;
        box_.blurSingleBox(recMat, sigma, ksize, unif); 
        }
        break;
    default:
        cout << "Noise model not implemented" << endl;
        exit(EXIT_FAILURE);
        break;
    }
}        
        
void EvaluateMetricsOnCuboid::evaluateCubOnNoiseModel(const vector<int>& indexes, int i) {
    //We create a reconstruction from ground truth
    cv::Mat_<uint8_t> recMat = cuboidGtMat_.clone();
    int j=0;
    //compute metric and save
    int sv_index = i + gaussian_index_;
    cuboid_all_results_.at(j).results.at(sv_index) = calcMetrics(cuboidGtMat_, recMat);
    while (j<1000) {
        //increment change
        //compute metric and save
        addNoise(indexes[j], NoiseModel(i), recMat);
        cuboid_all_results_.at(j).results.at(sv_index) = calcMetrics(cuboidGtMat_, recMat);
        j++;
        if (NoiseModel(i) == NoiseModel::GAUSSIAN) {
            //if all proba are within -0.4 / 0.6, we break
            vector<double> rec_vector;
            box_.getProbaVectorFromSingleBox(recMat, rec_vector);
            if (!box_.isBoxObserved(rec_vector, 0.1)){
                break;
            }
            
        }

    }
}

void EvaluateMetricsOnCuboid::evaluate() {
    //First, we need to check that the file does not already exists (xp already started)
    std::string fname = cuboid_info_.cuboid_output_folder+ "started_"+ std::to_string(cuboid_info_.id);
    fs::path fpath(fname);
    if (!fs::exists(fpath)) {
        //First, we check that we can save the results at the end:
        ofstream file(fname, ios::out);
        if (!file) {
            cerr << "cannot open " << fname << endl;
            exit(EXIT_FAILURE);
        }
        file.close();
        std::cout << "Starting: " << fpath << std::endl;
        //We create a single time the list of the indexes to incrementally change GT
        static std::random_device rd;
        static std::default_random_engine engine(rd());
        std::vector<int> indexes;
        for (int i=0; i<1000; i++) {
            indexes.push_back(i);
        }
        std::shuffle(indexes.begin(), indexes.end(), engine);
        
        for (int i=0; i<N_NOISE_MODEL; i++) {
            if (NoiseModel(i)==NoiseModel::GAUSSIAN) {
                continue;
            } else {
                gaussian_index_ = 0;
                evaluateCubOnNoiseModel(indexes, i);
                saveToDiskSelectedNoiseModel(i);
            }
        }
    }
}

//Compute the metrics
ComparatorDatatypes::Metrics EvaluateMetricsOnCuboid::calcMetrics(const cv::Mat_<uint8_t>& gtMat, const cv::Mat_<uint8_t>& recMat) const {

    ComparatorDatatypes::BoxToProcess box;
    cv::Range full = cv::Range(0,box_size_ );
    box.ranges = std::vector{full, full, full};
    box.metrics = ComparatorDatatypes::Metrics();
    vector<double> rec_vector;
    vector<double> gt_vector;
    box_.getProbaVectorFromSingleBox(gtMat, gt_vector);
    box_.getProbaVectorFromSingleBox(recMat, rec_vector);
    box.metrics.n_gt_points = ComputeMetrics::getNpoints(gt_vector, occupancy_thres_);    
    box.metrics.n_rec_points = ComputeMetrics::getNpoints(rec_vector, occupancy_thres_);    

    bool is_observed = box_.isBoxObserved(rec_vector, divergence_to_unknown_thres_);
    bool is_gt_empty = true;
    if (box.metrics.n_gt_points > 0) is_gt_empty = false;
    if (is_observed) {
        box.metrics.volumetric_information = ComputeMetrics::getVolumetricInformation(rec_vector);        
        //We do that for all 4 couples (likelihood, registration distance) defined
        //coverage 1
        int i = 0;
        box.metrics.n_match_1 = ComputeMetrics::getBoxNMatch(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.surface_coverage_1 = ComputeMetrics::getSurfaceCoverage(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.reconstruction_accuracy_1 = ComputeMetrics::getReconstructionAccuracy(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        //Average Hausdorff Distance and Kappa:
        box.metrics.ahd_1 = ComputeMetrics::getAverageHausdorffDistance(box,gtMat, recMat,
                    cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.kappa_1 = ComputeMetrics::getKappa(box,gtMat, recMat,
                    cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        //coverage 2
        i = 1;
        box.metrics.n_match_2 = ComputeMetrics::getBoxNMatch(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.surface_coverage_2 = ComputeMetrics::getSurfaceCoverage(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.reconstruction_accuracy_2 = ComputeMetrics::getReconstructionAccuracy(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        //coverage 3
        i = 2;
        box.metrics.n_match_3 = ComputeMetrics::getBoxNMatch(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.surface_coverage_3 = ComputeMetrics::getSurfaceCoverage(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.reconstruction_accuracy_3 = ComputeMetrics::getReconstructionAccuracy(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        //Average Hausdorff Distance and Kappa:
        box.metrics.ahd_3 = ComputeMetrics::getAverageHausdorffDistance(box,gtMat, recMat,
                    cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.kappa_3 = ComputeMetrics::getKappa(box,gtMat, recMat,
                    cov_thresholds[i].likelihood,img_width_, img_height_, box_size_, img_res_);        
        //coverage 4
        i = 3;
        box.metrics.n_match_4 = ComputeMetrics::getBoxNMatch(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.surface_coverage_4 = ComputeMetrics::getSurfaceCoverage(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);        
        box.metrics.reconstruction_accuracy_4 = ComputeMetrics::getReconstructionAccuracy(box,gtMat, recMat,
                    cov_thresholds[i].reg_dist, cov_thresholds[i].likelihood, img_width_, img_height_, box_size_, img_res_);


    } else {
        box.metrics.n_match_1 = 0;        
        box.metrics.n_match_2 = 0;        
        box.metrics.n_match_3 = 0;        
        box.metrics.n_match_4 = 0;        
        box.metrics.volumetric_information = non_observed_vi_;        
        box.metrics.surface_coverage_1 = non_observed_cov_;
        box.metrics.surface_coverage_2 = non_observed_cov_;
        box.metrics.surface_coverage_3 = non_observed_cov_;
        box.metrics.surface_coverage_4 = non_observed_cov_;
        box.metrics.reconstruction_accuracy_1 = non_observed_acc_;
        box.metrics.reconstruction_accuracy_2 = non_observed_acc_;
        box.metrics.reconstruction_accuracy_3 = non_observed_acc_;
        box.metrics.reconstruction_accuracy_4 = non_observed_acc_;
        box.metrics.ahd_1 = box_size_*sqrt(3)*img_res_;
        box.metrics.ahd_3 = box_size_*sqrt(3)*img_res_;
        box.metrics.kappa_1 = -1;
        box.metrics.kappa_3 = -1;

    }
    if (is_gt_empty) {
        if (is_observed) {
            box.metrics.emptyness_l1 = ComputeMetrics::getEmptynessL1(rec_vector);
        } else {
            box.metrics.emptyness_l1 = non_observed_l1_;
        }
    }
    if (is_observed) {
        box.metrics.dkl = ComputeMetrics::getKLDivergence(rec_vector, gt_vector);
    } else {
        box.metrics.dkl = non_observed_dkl_;
    }
    if (!is_gt_empty) {
        if (is_observed) {
            ComputeOTMetrics::OccFreeWasserstein wasserstein;
            wasserstein = ot_metrics_.normalizeAndGetSinkhornDistanceSigned(rec_vector, gt_vector);
            box.metrics.wasserstein_occ_remapped = wasserstein.occ;
            box.metrics.wasserstein_free_remapped = wasserstein.free;
            box.metrics.wasserstein_direct = ot_metrics_.normalizeAndGetSinkhornDistance(rec_vector, gt_vector);
        } else {
            box.metrics.wasserstein_occ_remapped = non_observed_wd_;
            box.metrics.wasserstein_free_remapped = non_observed_wd_;
            box.metrics.wasserstein_direct = non_observed_wd_;

        }
    }
    return box.metrics;
}

void EvaluateMetricsOnCuboid::saveToDiskSelectedNoiseModel(int j) const {
        //Store one file per noise model, one line per iteration
        std::string fname = cuboid_info_.cuboid_output_folder + "results_increment" + std::to_string(cuboid_info_.id) + "_" + std::to_string(j) + ".csv";
        ofstream file(fname, ios::out);
        if (!file) {
            cerr << "cannot open " << fname << endl;
            exit(EXIT_FAILURE);
        }
        file << "i"
                << " dkl"
                << " wasserstein_occ_remapped wasserstein_free_remapped wasserstein_direct"
                << " surface_coverage_1 surface_coverage_2 surface_coverage_3 surface_coverage_4"
                << " reconstruction_accuracy_1 reconstruction_accuracy_2 reconstruction_accuracy_3 reconstruction_accuracy_4" 
                << " volumetric_information"
                << " n_match_1 n_match_2 n_match_3 n_match_4"
                << " ahd_1 ahd_3 kappa_1 kappa_3"
                << " emptyness_l2 emptyness_l1 n_rec_points n_gt_points"
                << endl;
        for (int i=0; i<1000; i++) {
            //EvaluateMetricsOnCuboid::CuboidResults& c_res = cuboid_all_res.at(i).results.at(j);
            const ComparatorDatatypes::Metrics& metrics = cuboid_all_results_.at(i).results.at(j);
            file << setprecision(0) << i;
            file << setprecision(8);
            file << " " << metrics.dkl;
            file << " " << metrics.wasserstein_occ_remapped << " " << metrics.wasserstein_free_remapped << " " << metrics.wasserstein_direct;
            file << " " << metrics.surface_coverage_1 << " " << metrics.surface_coverage_2 << " " << metrics.surface_coverage_3 << " " << metrics.surface_coverage_4
                        << " " << metrics.reconstruction_accuracy_1  << " " << metrics.reconstruction_accuracy_2 << " " << metrics.reconstruction_accuracy_3 << " " << metrics.reconstruction_accuracy_4 
                        << " " << metrics.volumetric_information;
            file << " " << metrics.n_match_1 << " " << metrics.n_match_2 << " " << metrics.n_match_3 << " " << metrics.n_match_4 ;
            file << " " << metrics.ahd_1 << " " << metrics.ahd_3 << " " << metrics.kappa_1 << " " << metrics.kappa_3;
            file << " " << metrics.emptyness_l2 << " " << metrics.emptyness_l1; 
            file << " " << metrics.n_rec_points << " " << metrics.n_gt_points; 
            file << endl;
        }
        file.close();
}

void EvaluateMetricsOnCuboid::saveToDisk() const {
    for (int k=0; k<N_NOISE_MODEL; k++) {
        saveToDiskSelectedNoiseModel(k);
        if (k == N_NOISE_MODEL -1) {
            for (int j=0; j< N_GAUSSIAN_BLUR; j++) {
                saveToDiskSelectedNoiseModel(k+j);
            }
        }

    }
}

//Multithreading
static list<EvaluateMetricsOnCuboid::CuboidInfo> cuboids_to_process; 
static mutex cuboids_mutex; 

static void processListOfCuboids() {
    while (true) {
        EvaluateMetricsOnCuboid::CuboidInfo c;
        {
            lock_guard<mutex> lock(cuboids_mutex);
	        if (cuboids_to_process.empty()) {
		        break;
	        }
	        c = cuboids_to_process.front();
            cuboids_to_process.pop_front();
            cout << "Starting new cuboid. " << cuboids_to_process.size() << " cuboid remaining." << endl;
        }
        try {
            EvaluateMetricsOnCuboid c_eval(c);
            c_eval.evaluate();
        }
        catch (...) {
            cerr << "Error processing cuboid in folder: " << c.cuboid_img_folder << ", id: " << c.id << endl;
        }
    }
}

int main(int argc,char * argv[]) {

    std::string fname; 
    std::string base_dir; 
    std::string result_name; 
    int num_threads = 4; 
    if (argc>2) {
        fname = std::string(argv[1]);
        base_dir =  std::string(argv[2]);
        result_name =  std::string(argv[3]);
        if (argc>3) {
            num_threads = atoi(argv[4]);
        }
    } else {
        printf("Usage: %s <cuboids_to_process completepath> <base_dir> <result_name> <option: num_thread>\n",argv[0]);
        return -1;
    }
    
    //The list of cuboids is expected as follows:
    //xpname cuboid_id img_res
    //This information is loaded in cuboids_to_process
    ifstream file(fname, ios::in);
    if (!file) {
        cerr << "cannot open " << fname << endl;
        exit(EXIT_FAILURE);
    }
    std::vector<EvaluateMetricsOnCuboid::CuboidInfo> cuboids_to_process_vector; 
    EvaluateMetricsOnCuboid::CuboidInfo c_cuboid;
    while (file >> c_cuboid.xp >> c_cuboid.id >> c_cuboid.res) {
        c_cuboid.cuboid_img_folder = base_dir + "/res/obs/" + c_cuboid.xp + "/cuboids/";
        c_cuboid.cuboid_output_folder = c_cuboid.cuboid_img_folder + "/" + result_name + "/";
        //cuboids_to_process_vector.push_back(c_cuboid);
        cuboids_to_process.push_back(c_cuboid);
    }
    file.close();
    //Process the cuboids 
    vector<shared_ptr<thread> > threads(num_threads);
    for (int i=0; i<num_threads; i++) {
        threads[i].reset(new thread(&processListOfCuboids));
    }
    for (int i=0; i<num_threads; i++) {
        threads[i]->join();
    }

    return 0;
}
