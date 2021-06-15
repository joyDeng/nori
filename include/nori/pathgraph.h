#pragma once
#include "common.h"
#include "color.h"
#include <nori/vector.h>
#include <string>
#include "nori/shadingPoint.h"


using nori::NoriException;
using nori::NoriObjectFactory;
using nori::Point2f;
using nori::Point2i;
using nori::Point3f;
using nori::Color3f;

typedef struct pathVertex{
    Color3f eLi;
    Point3f pos;
    Point3f dir;
    Point3f nor;
} pVertex;

typedef struct CompleteLightPath {
    int xIdx;
    int yIdx;
    size_t firstPathPointIdx;
    size_t numOfPathPoints;
    Color3f em;
} cPath;

typedef struct aabbinfo {
        Point3f min;
        Point3f max;
        Point3f center;
        Point3f extents;
        int longAxis;
        int shortAxis;
    } AABBINFO;

class PathGraph {
    public:
    PathGraph(){m_spCount = 0;};
    PathGraph(std::string filename) {loadGraph(filename);};
    ~PathGraph() {
        std::cout<<"deconstructor 1"<<std::endl;
        if(m_eigenvector_allocated) delete m_eigenvector;
        if(m_cpl_allocated) delete m_cpl;
        if(m_max_idx_allocated) delete m_max_idx;
        std::cout<<"deconstructor 2"<<std::endl;
    };

    bool loadShadingPoints(std::string foldername);
    bool loadLightPoints(std::string foldername);
    bool loadPaths(std::string foldername);
    bool loadAABB(std::string foldername);
    bool loadGraph(std::string foldername);
    bool loadNeighbors(std::string foldername);
    void computeDimensions(AABBINFO aabb, const int N);


    SPoints m_sps;
    
    Eigen::Matrix4f m_camera_matrix, m_camera2sample;
    float * m_eigenvector;
    
    cPath *m_cpl;
    AABBINFO m_aabb;
    size_t m_spCount;
    size_t m_pathCount;
    int m_xresolution;
    int m_yresolution;
    int *m_max_idx;

    bool m_max_idx_allocated = false;
    bool m_cpl_allocated = false;
    bool m_eigenvector_allocated = false;

    float m_fov;
    float m_near_clip;
};