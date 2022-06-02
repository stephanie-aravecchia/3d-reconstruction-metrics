#include "slice_octomap/slice_octomap_node.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace octomap;


SliceOctomap::SliceOctomap(double resolution, double occ_thres, std::string base_dir, std::string xp) : 
        resolution_(resolution), occ_threshold_(occ_thres) {
    outdir_ = base_dir + "3d_grid/img/" + xp + "/";  
    string fname = base_dir + "3d_grid/ot/" + xp + "/" + xp + ".ot";
    cout << "outdir set to: " << outdir_ << endl;
    cout << "loading octree " << fname << "\n";
    tree_ = dynamic_cast<OcTree*>(OcTree::read(fname));
    tree_->getMetricMax(bbox_.xmax, bbox_.ymax, bbox_.zmax);
    tree_->getMetricMin(bbox_.xmin, bbox_.ymin, bbox_.zmin);
    cout << "BBOX is set automatically\n";
    cout <<  "BBox: "
        << bbox_.xmin << " "<< bbox_.ymin << " "<< bbox_.zmin << " "
        << bbox_.xmax << " "<< bbox_.ymax << " "<< bbox_.zmax << endl;

    cout << "Octree has " << tree_->size() << " nodes.\n";
    initializeMatVector();
    saveBBoxFile();
}

SliceOctomap::SliceOctomap(double resolution, double occ_thres, const ComparatorDatatypes::BBox& bbox,
            std::string base_dir, std::string xp) : 
        resolution_(resolution), occ_threshold_(occ_thres) {
    outdir_ = base_dir + "3d_grid/img/" + xp + "/";  
    string fname = base_dir + "3d_grid/ot/" + xp + "/" + xp + ".ot";
    cout << "outdir set to: " << outdir_ << endl;
    cout << "loading octree " << fname << "\n";
    tree_ = dynamic_cast<OcTree*>(OcTree::read(fname));
    tree_->getMetricMax(bbox_.xmax, bbox_.ymax, bbox_.zmax);
    tree_->getMetricMin(bbox_.xmin, bbox_.ymin, bbox_.zmin);
    cout <<  "initial bbox: "
        << bbox_.xmin << " "<< bbox_.ymin << " "<< bbox_.zmin << " "
        << bbox_.xmax << " "<< bbox_.ymax << " "<< bbox_.zmax << endl;
    bbox_.xmin = bbox.xmin; 
    bbox_.xmax = bbox.xmax; 
    bbox_.ymin = bbox.ymin; 
    bbox_.ymax = bbox.ymax; 
    bbox_.zmin = bbox.zmin; 
    bbox_.zmax = bbox.zmax; 
    cout <<  "Selected bbox: "
        << bbox_.xmin << " "<< bbox_.ymin << " "<< bbox_.zmin << " "
        << bbox_.xmax << " "<< bbox_.ymax << " "<< bbox_.zmax << endl;
    cout << "BBOX is set manually from user input\n";
    cout << "Complete octree has " << tree_->size() << " nodes.\n";
    initializeMatVector();
    saveBBoxFile();
}
SliceOctomap::SliceOctomap(double resolution, double occ_thres, double border,
            std::string base_dir, std::string xp) : 
        resolution_(resolution), occ_threshold_(occ_thres) {
    outdir_ = base_dir + "3d_grid/img/" + xp + "/";  
    string fname = base_dir + "3d_grid/ot/" + xp + "/" + xp + ".ot";
    cout << "outdir set to: " << outdir_ << endl;
    cout << "loading octree " << fname << "\n";
    tree_ = dynamic_cast<OcTree*>(OcTree::read(fname));
    tree_->getMetricMax(bbox_.xmax, bbox_.ymax, bbox_.zmax);
    tree_->getMetricMin(bbox_.xmin, bbox_.ymin, bbox_.zmin);
    cout <<  "initial bbox: "
        << bbox_.xmin << " "<< bbox_.ymin << " "<< bbox_.zmin << " "
        << bbox_.xmax << " "<< bbox_.ymax << " "<< bbox_.zmax << endl;
    bbox_.xmin += border; 
    bbox_.ymin += border; 
    bbox_.xmax -= border; 
    bbox_.ymax -= border; 
    cout <<  "new bbox: "
        << bbox_.xmin << " "<< bbox_.ymin << " "<< bbox_.zmin << " "
        << bbox_.xmax << " "<< bbox_.ymax << " "<< bbox_.zmax << endl;
    cout << "BBOX is set with a manual offset\n";
    cout << "Complete octree has " << tree_->size() << " nodes.\n";
    initializeMatVector();
    saveBBoxFile();
}

void SliceOctomap::saveBBoxFile() const {
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


void SliceOctomap::initializeMatVector() {
    size_t n{static_cast<size_t>((bbox_.zmax - bbox_.zmin)/resolution_)+1};
    img_size_ = cv::Size(static_cast<int>((bbox_.xmax - bbox_.xmin)/resolution_)+1, static_cast<int>((bbox_.ymax - bbox_.ymin)/resolution_)+1);
    for (size_t i{0}; i < n ; i++) {
        //we want to keep the full probability map, with the unknown voxels:
        if (occ_threshold_ <= 1e-2) {
            mat_vector_.push_back(cv::Mat_<uint8_t>(img_size_,127));//(probability .5 corresponds to unknown)
        } else { //we want only the occupied cells:
            mat_vector_.push_back(cv::Mat_<uint8_t>(img_size_,0));
        }
    }
}

void SliceOctomap::saveSliceImgs() const {
    for (int i{0}; i < static_cast<int>(mat_vector_.size()) ; i++) {
        char fname[30];
        sprintf(fname, "otslice_%04d.png", i);
        cv::imwrite(outdir_ + fname, mat_vector_[i]);
    }
}

uint8_t SliceOctomap::getNormalizedOccupancy(double v) const {
    return static_cast<uint8_t>(std::max(0., std::min((v*255), 255.)));
}

std::array<size_t, 2> SliceOctomap::getSliceIndexes(const SliceOctomap::BBox& vbox) const {
    return std::array<size_t, 2>{static_cast<size_t>(std::max(0.,std::floor((vbox.zmin-bbox_.zmin)/resolution_))),
        static_cast<size_t>(std::min(std::ceil((vbox.zmax-bbox_.zmin)/resolution_),(double) mat_vector_.size()))};
}
void SliceOctomap::bboxToPix(const SliceOctomap::BBox& vbox, SliceOctomap::PixBBox& pixbox) const {
    pixbox.xmin = round(min(max(0.,(vbox.xmin-bbox_.xmin)/resolution_),static_cast<double>(img_size_.width-1)));
    pixbox.xmax = round(max(0.,min((vbox.xmax-bbox_.xmin)/resolution_-1,static_cast<double>(img_size_.width-1))));
    pixbox.ymin = round(min(max(0.,(vbox.ymin-bbox_.ymin)/resolution_),static_cast<double>(img_size_.height-1)));
    pixbox.ymax = round(max(0., min((vbox.ymax-bbox_.ymin)/resolution_-1,static_cast<double>(img_size_.height-1))));
    pixbox.zmin = round(min(max(0.,(vbox.zmin-bbox_.zmin)/resolution_),static_cast<double>(mat_vector_.size()-1)));
    pixbox.zmax = round(max(0., min((vbox.zmax-bbox_.zmin)/resolution_-1,static_cast<double>(mat_vector_.size()-1))));
}

void SliceOctomap::putOccupiedLeavesInImg() {
    //iterate over all the leafs,
    cout << "Starting to slice the octree in " << mat_vector_.size() << endl;
    point3d bboxMin(bbox_.xmin, bbox_.ymin, bbox_.zmin);
    point3d bboxMax(bbox_.xmax, bbox_.ymax, bbox_.zmax);
    for(OcTree::leaf_bbx_iterator it = tree_->begin_leafs_bbx(bboxMin,bboxMax),
                end=tree_->end_leafs_bbx(); it!= end; ++it) {
        //keep only the voxels occupied above a threshold:
        if (it->getOccupancy() >= occ_threshold_) {
            //normalize the occupancy to use as a color                 
            uint8_t color{getNormalizedOccupancy(it->getOccupancy())};
            //get the bbox of the voxel, and the indices of the cv::Mat in which to draw the rectangles, and draw them
            BBox vBBox{it.getCoordinate(), it.getSize()};
            //getCoordinates is the center
            PixBBox pixBBox;
            bboxToPix(vBBox, pixBBox);
            for (size_t i{pixBBox.zmin}; i < pixBBox.zmax+1; ++i) {
                assert(pixBBox.xmin >= 0);
                assert(pixBBox.xmin < img_size_.width);
                assert(pixBBox.xmax >= 0);
                assert(pixBBox.xmax < img_size_.width);
                assert(pixBBox.ymin >= 0);
                assert(pixBBox.ymin < img_size_.height);
                assert(pixBBox.ymax >= 0);
                assert(pixBBox.ymax < img_size_.height);
                assert(i >= 0); 
                assert(i < mat_vector_.size());
                //avoid vertical or horizontal lines artefacts:
                if ((pixBBox.xmax == pixBBox.xmin) || (pixBBox.ymin == pixBBox.ymax)) {
                    cv::rectangle(mat_vector_[i],cv::Point(pixBBox.xmin, pixBBox.ymin), 
                        cv::Point(pixBBox.xmax, pixBBox.ymax),cv::Scalar(color), 1);
                } else {
                    cv::rectangle(mat_vector_[i],cv::Point(pixBBox.xmin, pixBBox.ymin), 
                        cv::Point(pixBBox.xmax, pixBBox.ymax),cv::Scalar(color), -1);
                }
            }
        }
    }
}
void SliceOctomap::slice() {
    putOccupiedLeavesInImg();
    cout << "Slicing of the octree is complete, starting to save the images to disk in " << outdir_ << endl;
    saveSliceImgs();
}

int main(int argc, char** argv) {

    if (argc < 6) {
        std::cerr << "Invalid number of arguments : rosrun slice_octomap slice_octomap_node BASE_DIR XP_NAME resolution occupancy_threshold xmin xmax ymin ymax zmin zmax\n";
        exit(EXIT_FAILURE);
    }
    std::string base_dir = std::string(argv[1]);
    std::string xp = std::string(argv[2]);
    double resolution = atof(argv[3]);
    double occ_threshold = atof(argv[4]);
    ComparatorDatatypes::BBox bbox{atof(argv[5]), atof(argv[6]), atof(argv[7]), atof(argv[8]), atof(argv[9]), atof(argv[10])};
    //if provided bbox is 0 0 0 0 0 0 just slice the complete octree
    if (fabs(bbox.xmin-bbox.xmax)<=1e-6) {
        SliceOctomap slicer(resolution, occ_threshold, base_dir, xp);
        slicer.slice();
    } else {
        SliceOctomap slicer(resolution, occ_threshold, bbox, base_dir, xp);
        slicer.slice();
    }
};
