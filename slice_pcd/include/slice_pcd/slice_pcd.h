#ifndef SLICE_PCD_H
#define SLICE_PCD_H
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <array>
#include "comparator/dataset_info.h"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

//This class Slice a PCD into images
class SlicePCD {
    protected:
        pcl::PointCloud<pcl::PointXYZ> cloud_; 
        std::string fname_;
        std::string outdir_;
        std::vector<cv::Mat_<uint8_t> > mat_vector_;
        double resolution_;
        cv::Size img_size_;

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
        struct PixBBox {
            unsigned int xmin;
            unsigned int xmax;
            unsigned int ymin;
            unsigned int ymax;
            unsigned int zmin;
            unsigned int zmax;
            PixBBox() {}
            PixBBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax):
                xmin(static_cast<unsigned int>(xmin)), xmax(static_cast<unsigned int>(xmax)), 
                ymin(static_cast<unsigned int>(ymin)), ymax(static_cast<unsigned int>(ymax)), 
                zmin(static_cast<unsigned int>(zmin)), zmax(static_cast<unsigned int>(zmax)) {}
        };
        
        struct Pix {
            int x;
            int y;
            int z;
            Pix() {}
            Pix(int x, int y, int z): x(x), y(y), z(z) {}
        };
        
        BBox bbox_;
        //initialize a vector of cv::Mat, size of the vector depend on the height of the octree and the resolution,
        //size of the cv::Mat depends on the lenght and widht of the octree and the resolution
        void initializeMatVector();
        //iterate over all the points,draw a rectange in the appropriate cv::Mat
        void putPointsInImg();
        //get the indices of the slice images in which to draw the rectangles, from the height of the voxel
        std::array<size_t, 2> getSliceIndexes(const BBox&) const;
        Pix pointToPix(const pcl::PointXYZ&) const;
        bool isPointInGrid(const pcl::PointXYZ&) const;
        //save the created slice images to disk
        void saveSliceImgs() const;
        void saveBBoxFile() const;

    public:
        //Constructor with a target BBOX to constrain the slicer to this BBOX
        SlicePCD(double, const ComparatorDatatypes::BBox&, std::string, std::string);
        //slice the pcd
        void slice();
};

#endif