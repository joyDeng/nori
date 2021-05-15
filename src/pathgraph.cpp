#include "nori/pathgraph.h"
#include <nori/shadingPoint.h>
#include <iostream>
#include <fstream>
using namespace std;

// need to use absolute path here
bool PathGraph::loadShadingPoints(std::string foldername){
    bool ret = false;
    fstream graphfile;
    std::cout<<"[loadGraph] Open successfully, Reading from "<<foldername<<std::endl;
    std::string filename = foldername + "_vert.bin";
    graphfile.open(filename.c_str(), ios::in | ios::binary);

    if(graphfile.is_open()) {
        std::cout<<"[loadGraph] Open successfully, Reading file "<<filename<<std::endl;
        graphfile.seekg(0);
        graphfile.read((char *)&m_spCount, sizeof(int));
        std::cout<<"[loadGraph] m_spCount =  "<<m_spCount<<std::endl;

        m_sps.allocateHostSpace(m_spCount);
        graphfile.read((char *)m_sps.h_sps, sizeof(SPoint) * m_spCount);


        std::cout<<"[loadGraph] Second Element = {eLi: ="<<Point3f(m_sps.h_sps[1].eLi[0], m_sps.h_sps[1].eLi[1], m_sps.h_sps[1].eLi[2]).toString()
                <<" nor: ="<<Point3f(m_sps.h_sps[1].geoN[0], m_sps.h_sps[1].geoN[1], m_sps.h_sps[1].geoN[2]).toString()
                <<" pos: ="<<Point3f(m_sps.h_sps[1].pos[0], m_sps.h_sps[1].pos[1], m_sps.h_sps[1].pos[2]).toString()
                <<" dir: ="<<Point3f(m_sps.h_sps[1].wi_d[0], m_sps.h_sps[1].wi_d[1], m_sps.h_sps[1].wi_d[2]).toString()<<std::endl;

        graphfile.close();
        ret = true;
    }else{
        std::cout<<"[loadGraph] Failed to open file: "<<filename<<std::endl;
        ret = false;
    }

    return ret;
}

void PathGraph::computeDimensions(AABBINFO aabb, const int N){
    Point3f extents = aabb.extents;
    Point3f ratio = extents / extents[aabb.shortAxis];
    float dim = pow(N, 1.0 / 3.0)+1.0;
    Point3f dimensions = ratio * dim + Point3f(1.0f);
    int grid_resolution[3] = {(int)dimensions[0], (int)dimensions[1], (int)dimensions[2]};
    memcpy(m_sps.my_dim, grid_resolution, 3 *sizeof(int));
    printf("resolution of the hash grid is [%d x %d x %d]", grid_resolution[0], grid_resolution[1], grid_resolution[2]);
    printf("size of the hash grid is [%.3f x %.3f x %.3f]", extents[0], extents[1], extents[2]);
    printf("ratio of the hash grid is [%.3f x %.3f x %.3f]", ratio[0], ratio[1], ratio[2]);
}

bool PathGraph::loadPaths(std::string foldername){
    bool ret = true;
    std::string filename = foldername + "_paths.bin";
    fstream pathfile;
    pathfile.open(filename.c_str(), ios::in | ios::binary);

    if(pathfile.is_open()) {
        std::cout<<"[loadGraph] Open successfully, Reading file "<<filename<<std::endl;
        pathfile.seekg(0);
        pathfile.read((char *)&m_pathCount, sizeof(size_t));
        std::cout<<"[loadGraph] m_pathCount =  "<<m_pathCount<<std::endl;

        pathfile.read((char *)&m_xresolution, sizeof(int));
        pathfile.read((char *)&m_yresolution, sizeof(int));

        std::cout<<"[loadGraph] resolution = {pathnumber = "<< m_pathCount 
                << " x = "<<m_xresolution<<" y ="<<m_yresolution<<std::endl;

        m_cpl = new cPath [m_pathCount];
        m_cpl_allocated = true;
        pathfile.read((char *)m_cpl, sizeof(cPath) * m_pathCount);

        std::cout<<"[loadGraph] Second Element = {xIdx: ="<<m_cpl[1].xIdx
                <<" yIdx: ="<<m_cpl[1].yIdx
                <<" numPathPoint: ="<<m_cpl[1].numOfPathPoints
                <<" firstPathPointIdx: ="<<m_cpl[1].firstPathPointIdx<<std::endl;

        pathfile.close();
        ret &= true;
    }else{
        std::cout<<"[loadGraph] Failed to open file: "<<filename<<std::endl;
        ret = false;
    }
    return ret;
}

bool PathGraph::loadLightPoints(std::string foldername){
    bool ret = true;
    std::string filename = foldername + "_light.bin";
    fstream lightfile;
    lightfile.open(filename.c_str(), ios::in | ios::binary);

    if(lightfile.is_open()){
        lightfile.seekg(0);
        int numlight;
        lightfile.read((char *)&numlight, sizeof(int));
        std::cout<<"[loadGraph] numlight =  "<<numlight<<"num shadingpoint: = "<< m_spCount<<std::endl;
       
        lightfile.read((char *)m_sps.h_lps, sizeof(LPoint) * numlight);
        std::cout<<"[loadGraph] Second Element = {Lem: ="<<Point3f(m_sps.h_lps[1].L_em[0], m_sps.h_lps[1].L_em[2], m_sps.h_lps[1].L_em[3]).toString()<<std::endl;
        lightfile.close();
        ret &= true;
    }else{
        std::cout<<"[loadGraph] Failed to open file: "<<filename<<std::endl;
        ret = false;
    }

    return ret;
}

bool PathGraph::loadAABB(std::string foldername){
    bool ret = true;
    std::string filename = foldername + "_aabb.bin";
    fstream aabbfile;
    aabbfile.open(filename.c_str(), ios::in | ios::binary);

    if(aabbfile.is_open()){
        aabbfile.seekg(0);
        aabbfile.read((char *)&m_aabb, sizeof(AABBINFO));
        memcpy(m_sps.my_minb, &m_aabb.min, 3 * sizeof(float));
        memcpy(m_sps.my_maxb, &m_aabb.max, 3 * sizeof(float));
        std::cout<<"[loadGraph] aabb =  "<<m_aabb.min<<"num shadingpoint: = "<< m_aabb.max<<std::endl;
        aabbfile.close();
        ret &= true;
    }else{
        std::cout<<"[loadGraph] Failed to open file aabb.bin: "<<filename<<std::endl;
        ret = false;
    }
    return ret;
}

bool PathGraph::loadGraph(std::string foldername){
    bool ret = loadShadingPoints(foldername);
    if (ret){
        ret &= loadPaths(foldername);
        ret &= loadLightPoints(foldername);
    }
    
    std::string filename = foldername + "_sensor.bin";
    fstream sensorfile;
    sensorfile.open(filename.c_str(), ios::in | ios::binary);

    if(sensorfile.is_open()){
        sensorfile.seekg(0);
        sensorfile.read((char *)&m_camera_matrix, sizeof(Eigen::Matrix4f));
        sensorfile.read((char *)&m_camera2sample, sizeof(Eigen::Matrix4f));
        sensorfile.read((char *)&m_fov, sizeof(float));
        sensorfile.read((char *)&m_near_clip, sizeof(float));
        m_camera_matrix.transposeInPlace();
        m_camera2sample.transposeInPlace();

        std::cout<<"[loadGraph] camera matrix \n"<<m_camera_matrix<<std::endl;
        std::cout<<"[loadGraph] camer2sample matrix \n"<<m_camera2sample<<std::endl;
        sensorfile.close();
        ret &= true;
    }else{
        std::cout<<"[loadGraph] Failed to open file: "<<filename<<std::endl;
        ret = false;
    }

    

    std::ostringstream stringStream;
    stringStream << foldername<<"_scene_output_d"<<m_spCount<<"_ev_01.bin";
    filename = stringStream.str();
    std::cout<<filename<<std::endl;
    fstream eigenvectorsfile;
    eigenvectorsfile.open(filename.c_str(), ios::in | ios::binary);
    std::cout<<filename<<std::endl;

    if(eigenvectorsfile.is_open()){
        eigenvectorsfile.seekg(0);
        m_eigenvector = new float [m_spCount];
        m_eigenvector_allocated = true;
        eigenvectorsfile.read((char *)m_eigenvector, m_spCount * sizeof(float));
        eigenvectorsfile.close();
        std::cout<<"first element: "<<m_eigenvector[0] * 100000<<" last elements "<<m_eigenvector[m_spCount-1] *  100000<<std::endl;
        ret &= true;
    }else{
        std::cout<<"[loadGraph] Failed to open file aabb.bin: "<<filename<<std::endl;
        ret = false;
    }

    std::ostringstream sStream;
    sStream << foldername<<"_scene_output_d"<<m_spCount<<"_max_idx.bin";
    filename = sStream.str();
    std::cout<<filename<<std::endl;
    fstream maxcFile;
    maxcFile.open(filename.c_str(), ios::in | ios::binary);
    std::cout<<filename<<std::endl;

    if(maxcFile.is_open()){
        maxcFile.seekg(0);
        m_max_idx = new int [6];
        m_max_idx_allocated = true;
        maxcFile.read((char *)m_max_idx, 6 * sizeof(int));
        for (int i = 0 ; i < 6 ; i++)
            std::cout<<m_max_idx[i]<<std::endl;
        maxcFile.close();
        std::cout<<"first element: "<<m_max_idx[0]<<std::endl;
        ret &= true;
    }else{
        std::cout<<"[loadGraph] Failed to open file _max_idx.bin: "<<filename<<std::endl;
        ret = false;
    }
   
    return ret;
}