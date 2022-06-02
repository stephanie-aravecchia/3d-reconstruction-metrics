#include "slice_pcd/slice_pcd.h"
#include <iostream>
#include <fstream>
using namespace std;

SlicePCD::SlicePCD(double resolution, const ComparatorDatatypes::BBox& bbox,
            std::string base_dir, std::string xp) : 
        resolution_(resolution) {
    outdir_ = base_dir + "ground_truth_stl/img/" + xp + "/";  
    string fname = base_dir + "ground_truth_stl/pcd/" + xp + ".pcd";
    cout << "outdir set to: " << outdir_ << endl;
    cout << "loading pcd " << fname << "\n";
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (fname, cloud_) == -1) {//* load the file
        cerr << "Couldn't read file " + fname << endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Loaded "
                << cloud_.size()
                << " data points from " << fname 
                << std::endl;
    bbox_.xmin = bbox.xmin; 
    bbox_.xmax = bbox.xmax; 
    bbox_.ymin = bbox.ymin; 
    bbox_.ymax = bbox.ymax; 
    bbox_.zmin = bbox.zmin; 
    bbox_.zmax = bbox.zmax; 
    cout << "BBOX is set manually from user input\n";
    initializeMatVector();
    saveBBoxFile();
}

void SlicePCD::saveBBoxFile() const {
    string boxfile = outdir_ + "bbox";
    ofstream file(boxfile, ios::out);
    if (!file) {
        cerr << "cannot open " << boxfile << endl;
        exit(EXIT_FAILURE);
    }
    file << "xmin ymin zmin xmax ymax zmax res\n"
        << bbox_.xmin << " "<< bbox_.ymin << " "<< bbox_.zmin << " "
        << bbox_.xmax << " "<< bbox_.ymax << " "<< bbox_.zmax << " " << resolution_ << endl;
    file.close();

}

//Initialize with zeros
void SlicePCD::initializeMatVector() {
    size_t n{static_cast<size_t>((bbox_.zmax - bbox_.zmin)/resolution_)+1};
    img_size_ = cv::Size(static_cast<int>((bbox_.xmax - bbox_.xmin)/resolution_)+1, static_cast<int>((bbox_.ymax - bbox_.ymin)/resolution_)+1);
    for (size_t i{0}; i < n ; i++) {
        mat_vector_.push_back(cv::Mat_<uint8_t>(img_size_,0));
    }
}

void SlicePCD::saveSliceImgs() const {
    for (int i{0}; i < static_cast<int>(mat_vector_.size()) ; i++) {
        char fname[30];
        sprintf(fname, "slice_%04d.png", i);
        cv::imwrite(outdir_ + fname, mat_vector_[i]);
    }
}

std::array<size_t, 2> SlicePCD::getSliceIndexes(const SlicePCD::BBox& vbox) const {
    return std::array<size_t, 2>{static_cast<size_t>(std::max(0.,std::floor((vbox.zmin-bbox_.zmin)/resolution_))),
        static_cast<size_t>(std::min(std::ceil((vbox.zmax-bbox_.zmin)/resolution_),(double) mat_vector_.size()))};
}

bool SlicePCD::isPointInGrid(const pcl::PointXYZ& point) const {
    if ((point.x >= bbox_.xmin) && (point.x<=bbox_.xmax)) {
        if ((point.y >= bbox_.ymin) && (point.y<=bbox_.ymax)) {
            if ((point.z >= bbox_.zmin) && (point.z<=bbox_.zmax)) {
                return true;
            }
        }
    }
    return false;
}

SlicePCD::Pix SlicePCD::pointToPix(const pcl::PointXYZ& point) const {
    SlicePCD::Pix pix;
    pix.x = round(min(max(0.,(point.x-bbox_.xmin)/resolution_),static_cast<double>(img_size_.width-1)));
    pix.y = round(min(max(0.,(point.y-bbox_.ymin)/resolution_),static_cast<double>(img_size_.height-1)));
    pix.z = round(min(max(0.,(point.z-bbox_.zmin)/resolution_),static_cast<double>(mat_vector_.size()-1)));
    return pix;
}

void SlicePCD::putPointsInImg() {
    //iterate over all the points, color the associated pixel in white
    cout << "Starting to slice the pointcloud in " << mat_vector_.size() << endl;
    for (const auto& point: cloud_) {
        if (isPointInGrid(point)) {
            Pix pix = pointToPix(point);
            assert(pix.x >= 0);
            assert(pix.x < img_size_.width);
            assert(pix.y >= 0);
            assert(pix.y < img_size_.height);
            assert(pix.z >= 0); 
            assert(pix.z < mat_vector_.size());
            mat_vector_[pix.z](cv::Point(pix.x, pix.y)) = 255;
        }
    }
}
void SlicePCD::slice() {
    putPointsInImg();
    cout << "Slicing of the pcd is complete, starting to save the images to disk in " << outdir_ << endl;
    saveSliceImgs();
}

int main(int argc, char** argv) {

    if (argc < 6) {
        std::cerr << "Invalid number of arguments : rosrun slice_pcd slice_pcd_node BASE_DIR XP_NAME resolution xmin xmax ymin ymax zmin zmax\n";
        exit(EXIT_FAILURE);
    }
    std::string base_dir = std::string(argv[1]);
    std::string xp = std::string(argv[2]);
    double resolution = atof(argv[3]);
    ComparatorDatatypes::BBox bbox{atof(argv[4]), atof(argv[5]), atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9])};
    SlicePCD slicer(resolution, bbox, base_dir, xp);
    slicer.slice();
};
