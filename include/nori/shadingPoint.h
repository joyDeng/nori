#pragma once
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <map>
#include <limits>
#include <cstdlib>
#include <algorithm>
#include <random>

#define DIM_GRID = 128

class HashGrid;
class SPoints;


typedef struct hashparameters{
    int dim[3];
    float cellsize[3];
    float minb[3];
    int k;
    int dk;
} hparam;

typedef struct GridEntry{
    float position[3] = {0.0f};
    int index;
} GridEntry;

typedef struct Cell{
    int offset;
} Cell;

typedef struct val3f{
    float value[3] = {0.0};
} val3f;

class HashGrid{
    public:
    std::vector<std::vector<GridEntry>> hashgrid;
    int number_of_grid = 0;
    std::vector<int> initIndex;
    int dimension[3]={0};
    float maxb[3] = {std::numeric_limits<float>::min()};
    float minb[3] = {std::numeric_limits<float>::max()};
    float cellsize[3] = {0.0f};
    int maxCellPerGrid = 0;
    int emptygrid = 0;

    HashGrid(){};

    HashGrid(int *dim, float *maxb, float *minb){
        initialize(dim, maxb, minb);
    }

    ~HashGrid(){
        number_of_grid = 0;
        hashgrid.clear();
        initIndex.clear();
    }

    bool initialize(int *dim, float *boundmax, float *boundmin){
        memcpy(dimension, dim, 3 * sizeof(int));
        memcpy(maxb, boundmax, 3 * sizeof(float));
        memcpy(minb, boundmin, 3 * sizeof(float));
        cellsize[0] = (maxb[0] - minb[0]) / (float)dimension[0];
        cellsize[1] =  (maxb[1] - minb[1]) / (float)dimension[1];
        cellsize[2] = (maxb[2] - minb[2]) / (float)dimension[2];
        number_of_grid = dim[0] * dim[1] * dim[2];
        std::cout<<"resize hash grid"<<number_of_grid<<std::endl;
        hashgrid.resize(number_of_grid);
        std::cout<<"resize Indices"<<number_of_grid<<std::endl;
        initIndex.resize(number_of_grid);
        return true;
    }

    bool Insert(float x, float y, float z, int index){
        int key = getKey(x, y, z);
        // hashgrid[key].push_back(GridEntry{{x, y, z}, index});
        return true;
    }

    int getKey(float x, float y, float z){
        int ix = std::max(std::min((int)floor((x - minb[0]) / cellsize[0]), dimension[0]-1),0);
        int iy = std::max(std::min((int)floor((y - minb[1]) / cellsize[1]), dimension[1]-1),0);
        int iz = std::max(std::min((int)floor((z - minb[2]) / cellsize[2]), dimension[2]-1),0);
        return  ix * dimension[1] * dimension[2] + iy * dimension[2] + iz;
    }

    void finalize(){
        int offset = 0;
        for (int i = 0 ; i < hashgrid.size() ; i++){
            initIndex[i] = offset;
            // initIndex[i].num = hashgrid[i].size();
            offset += hashgrid[i].size();
            if(hashgrid[i].size() > maxCellPerGrid)   
                maxCellPerGrid = hashgrid[i].size();
            if(hashgrid[i].size()== 0){
                emptygrid++;
            }
        }
    }

    void clear(){
        hashgrid.clear();
        initIndex.clear();
        maxCellPerGrid = 0;
        emptygrid = 0;
    }
};


typedef struct PDFsum{
    float value[100] = {-1.0};
} PDFsum;

typedef struct Neighborhood{
    int num;
    std::vector<std::vector<int>> index;
    std::vector<std::vector<float>> pdfsum;
    std::vector<float> pdfmarginal;
} Neighborhood;

struct ShadingPoint{
    float pos[3];
    float wi[3];
    float wi_d[3];
    float wo[3];
    float shN[3];
    float geoN[3];
    float diffuse[3];
    float specular[3];
    float eLi[3];
    float eLd[3];
    float roughness;
    float pdf;
    float rrpdf;
    int nidx = 0;
    int groupIdx;
    char bsdf_type='d';
};
typedef struct ShadingPoint SPoint;

struct LightPoint{
    float L_directsample[3];
    float L_bsdfsample[3];
    float L_em[3];
    float lightpdf;
    float bsdfpdf;
};
typedef struct LightPoint LPoint;

typedef struct ResultSpace{
    int iter=0;
    std::vector<val3f *> blur_results;
    std::vector<val3f *> mc_results;
    val3f * blur_direct;
} RadianceResult;


bool CUDAmalloc(void ** sp, size_t size_in_byte);
bool CUDAcpyH2D(void *device, void *host, size_t size_in_byte);
bool CUDAcpyD2H(void *device, void *host, size_t size_in_byte);
void CUDAdelete(void *device);
bool CUDAmemD2D(void *dst, void *src, size_t size_in_byte);
void CUDARadiance(SPoint *cu_sps, val3f *cu_result, int num);
bool CUDAmallocPitch(void **addr, size_t *pitch, size_t height_size_in_byte, size_t width);
void hashKNN(SPoint *cu_sps, GridEntry *cu_gridentries, int *cu_cellinfo, hparam hp, int *cu_neighbors, int sizeofsp, int *cu_knn);
void batchHashKNN(SPoint *cu_sps, GridEntry *cu_gridentries, int *cu_cellinfo, hparam hp, int *cu_neighbors, int sizeofsp, int maxCPG, int *cu_knn, int *cu_dknn, const size_t pitch);
void batchHashCluster(SPoint *cu_sps, GridEntry * gridentries, int *cellinfo, hparam hash_param, int *cu_clusters, int sizeofsp, int maxCPG, int sizeofcluster, int *cu_cluster_i, int *cu_clustermember_count);
void buildHashGridAndFindKNN(SPoint *cu_sps, hparam hp, int *cu_neighbors, int sizeofsp,int *cu_knn, int *cu_dknn, const size_t pitch_n);
/*compute all on gpu */
void computeMISAllOnGPU(const SPoint *cu_sps, const int * cu_neighbors, const int *cu_knn, const int *cu_dknn,
    int sizeofsp, int iter, int k, int dk, ResultSpace ret, const size_t pitch, bool jitter=false);
void computeMISAllOnGPURecord(const SPoint *cu_sps, const int * cu_neighbors, const int *cu_knn, int sizeofsp, int iter, int k, ResultSpace ret, const size_t pitch);
void computeScatterAllOnGPU(const SPoint *cu_sps, const int * cu_neighbors, const int *cu_knn, int sizeofsp, int iterations, int k, val3f *cu_radiance, const size_t pitch);
void computeScatterAllOnGPURecord(const SPoint *cu_sps, const int *cu_neighbors, const int *cu_knn, int sizeofsp, int iterations, int k, ResultSpace &result, const size_t pitch);
/*with direct light opt*/
void computeDirectScatterAllOnGPUWithWeight(const SPoint *cu_sps,const LPoint *cu_lps, const int *cu_neighbors, const int *cu_dknn, int sizeofsp, int k, val3f * cu_direct_radiance, const size_t pitch);
void computeDirectScatterAllOnGPU(const SPoint *cu_sps,const LPoint *cu_lps, const int *cu_neighbors, const int *cu_dknn, int sizeofsp, int k, val3f * cu_direct_radiance, const size_t pitch);
void computeScatterAllOnGPUWithDirectOpt(const SPoint *cu_sps,const int *cu_neighbors, const int *cu_knn, const int *cu_dknn, 
    int sizeofsp, int iterations, int k, int dk, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch, bool jitter=false);
void computeScatterAllOnGPURecordWithDirectOpt(const SPoint *cu_sps,const int *cu_neighbors, const int *cu_dknn, int sizeofsp, int iterations, int k, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch);
/*with direct light opt and weight*/
void computeScatterAllOnGPUWithDirectOptandWeight(const SPoint *cu_sps,const int *cu_neighbors, const int *cu_knn,const int *cu_dknn,
    int sizeofsp, int iterations, int k,  int dk, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch, bool jitter=false);
void computeScatterAllOnGPURecordWithDirectOptandWeight(const SPoint *cu_sps,const int *cu_neighbors, const int *cu_knn, int sizeofsp, int iterations, int k, val3f * cu_direct_radiance, ResultSpace ret, const size_t pitch);
/*with direct light opt cluster haven't check whether its working*/
void computeClusterScatterAllOnGPURecord(const SPoint *cu_sps, int numClusters, const int *cu_clusters, const int *cu_cluster_offset, int sizeofsp, int iterations, ResultSpace &ret);
void buildBatchClusters(SPoint *cu_sps, int sizeofsp, int tablecells, hparam hp, int numClusters, int * indices, int *cu_clusters, int *cu_np_in_clusters);
int countMyClusters(SPoint *cu_sps, int sizeofsp, hparam hp, int numClusters, int *cu_clusters, int *cu_np_in_clusters, int *cu_num_small_clusters);
void subClusters(SPoint *cu_sps, 
                int sizeofsp, 
                hparam hp, 
                int numOldClusters, 
                int numNewClusters, 
                int *cu_clusters, 
                const int *cu_clusters_old_offset, 
                int *cu_num_point_new_cluster,
                int *cu_num_small_clusters
                );
void FinalizeCluster(SPoint *cu_sps, int sizeofsp, hparam hp, int numClusters, int *cu_clusters, int *cu_np_in_clusters);                
void ClusterIterations(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const int num_of_clusters, const size_t *cu_element_offset, 
    int sizeofsp, const val3f * cu_elements, int iterations, ResultSpace &ret);
void ClusterIterations2(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const int num_of_clusters, const size_t *cu_element_offset, 
    int sizeofsp, const val3f * cu_elements,const val3f * cu_direct, int iterations, ResultSpace &ret);
void ClusterIterations3(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const int num_of_clusters, const size_t *cu_element_offset, 
    int sizeofsp, const val3f * cu_elements,const val3f * cu_direct_radiance, int iterations, ResultSpace &ret);
void precomputedMatrixElemtns(SPoint *cu_sps, const int *cu_clusters, const int *cu_cluster_offset, const size_t *cu_element_offset, const int num_of_clusters, const int sizeofsp, const int numzeroelemetns, val3f *cu_matrix_elements);
size_t MatrixElementsNumber(SPoint *cu_sps, const int *cu_offset, const int num_of_clusters, size_t * cu_elements_offset, int sizeofsp);
void ClusterDirect(const SPoint *cu_sps,const LPoint *cu_lps, const int *cu_clusters, const int *cu_clusters_offset, int sizeofsp, int num_of_clusters, val3f * cu_direct_radiance);
void ClusterScatterRecord(SPoint *cu_sps, int numClusters, const int *cu_clusters, 
    int *cu_np_in_clusters, int sizeofsp, int iterations, ResultSpace &ret);

class SPoints{
private:
    SPoint * cu_sps;
    LPoint * cu_lps;
    GridEntry *cu_grids;
    int *cu_cellinfo;
    int *cu_neighbors;
    int *cu_knn;
    int *cu_dknn;

    int *cu_clusters;
    // int *cu_clusters_i;
    int *cu_clusters_offsets;

    val3f min3f(float *a, val3f b){
        val3f t;
        for ( int i = 0 ; i < 3 ; i++)
            t.value[i] = std::min(a[i], b.value[i]);
        return t;
    };

    val3f max3f(float *a, val3f b){
        val3f t;
        for ( int i = 0 ; i < 3 ; i++)
            t.value[i] = std::max(a[i], b.value[i]);
        return t;
    };

public:
    /* data */
    SPoint * h_sps;
    LPoint * h_lps;
    int *h_clusters_offset;
    int *h_clusters;
    int num;
    int knn;
    int dknn;
    int maxK;
    int numberOfClusters;
    int numberOfPoints;
    bool hashostclusters = false;
    // int *h_clusters;
    Neighborhood m_neighbors;
    bool gpuMemAllocated = false;
    bool gpuMemUploaded = false;
    bool neighborhoodInitialized = false;
    bool hashgridbuilt = false;
    bool hashgriduploaded = false;
    bool neighborhoodcomputed = false;
    bool lightpointsinmemory = false;
    bool hasclusters = false;

    HashGrid m_grid;
    // val3f maxbound;
    // val3f minbound;
    ResultSpace m_result;

    float my_minb[3];
    float my_maxb[3];
    float my_cellsize[3];
    int my_dim[3];
    size_t pitch_n;

    void allocateResultSpaceForNiteration(int iteration){
        if(m_result.iter > 0){
            freeResultSpace();
        }
        
        m_result.iter = iteration;
        for(int i = 0 ; i < iteration ; i++){
            m_result.mc_results.push_back((val3f *) malloc(sizeof(val3f) * num));
            m_result.blur_results.push_back((val3f *) malloc(sizeof(val3f) * num));
        }
        m_result.blur_direct = (val3f *) malloc(sizeof(val3f) * num);
    }

    void freeResultSpace(){
        for(int i = 0 ; i < m_result.iter ; i++){
            free(m_result.mc_results[i]);
            free(m_result.blur_results[i]);
        }
        if(m_result.blur_direct != nullptr)
            free(m_result.blur_direct);
        m_result.mc_results.clear();
        m_result.blur_results.clear();
        m_result.iter = 0;
    }

    void freeClustersSpace(){
        if(hasclusters){
            CUDAdelete(cu_clusters);
            // CUDAdelete(cu_clusters_i);
            CUDAdelete(cu_clusters_offsets);
            hasclusters = false;
        }
    }

    
    /* from film storage to host local for vec3f */
    void put(SPoint *value, LPoint *light, int Idx){
        SPoint * adr = h_sps + Idx;
        LPoint * adrr = h_lps + Idx;

        memcpy(adr, value, sizeof(SPoint));
        memcpy(adrr, light, sizeof(LPoint));

        // maxbound = max3f(h_sps[Idx].pos, maxbound);
        // minbound = min3f(h_sps[Idx].pos, minbound);
    }
    
      void get(SPoint *value, LPoint *light, int Idx){
        SPoint * adr = h_sps + Idx;
        LPoint * adrr = h_lps + Idx;
        memcpy(value, adr, sizeof(SPoint));
        memcpy(light, adrr, sizeof(LPoint));
    }

    /* malloc space on host */
    bool allocateHostSpace(uint32_t N) {
        num = N;
        size_t size_in_byte = N * sizeof(SPoint);
        size_t size_in_lbyte = N * sizeof(LPoint);
        std::cout<<"allocating "<<size_in_byte<<" bytes to "<<N<<" shading points"<<std::endl;
        h_sps = (SPoint *)malloc(size_in_byte);
        h_lps = (LPoint *)malloc(size_in_lbyte);
        return true;
    }

    bool allocateNeighrbos(int cluster_num, int cluster_points_num){
        h_clusters = (int *)malloc(cluster_points_num * sizeof(int));
        h_clusters_offset = (int *)malloc(cluster_num * sizeof(int));
        hashostclusters = true;
        return true;
    }

    /* malloc space on device */
    bool allocateDeviceSpace(){
        if (!gpuMemAllocated){
            size_t size_in_bite = num * sizeof(SPoint);
            size_t size_in_Lbite = num * sizeof(LPoint);
            bool success = true;
            success &= CUDAmalloc((void **)&cu_sps, size_in_bite);
            success &= CUDAmalloc((void **)&cu_lps, size_in_Lbite);
            if(success) gpuMemAllocated = true;
            printf("\n device space allocated!");
            return success;
        }else{
            return true;
        }
    }

    void freelightpoints() {
        if(lightpointsinmemory){
            free(h_lps);
            lightpointsinmemory = false;
        }
    }

    /* free space on device */
    void freeDeviceSpace() {
        if(gpuMemAllocated) {
            CUDAdelete((void *)cu_sps);
            CUDAdelete((void *)cu_lps);
            gpuMemAllocated = false;
            gpuMemUploaded = false;
        }
        freeCudaHashGrid();
        freeCudaNeighbors();
    }

    void freeCudaNeighbors(){
        if(neighborhoodcomputed){
            CUDAdelete((void *)cu_neighbors);
            CUDAdelete(cu_knn);
            CUDAdelete(cu_dknn);
            neighborhoodcomputed = false;
        }
    }

    void freeCudaHashGrid(){
        if(hashgriduploaded){
            CUDAdelete((void *)cu_grids);
            CUDAdelete((void *)cu_cellinfo);
            hashgriduploaded = false;
        }
    }

    /* copy from device to host */
    bool cmp2Device(){
        size_t size_in_byte = num * sizeof(SPoint);
        size_t size_in_lbyte = num * sizeof(LPoint);
        bool success = true;
        if (gpuMemAllocated){
            success &= CUDAcpyH2D((void *)cu_sps, (void *)h_sps, size_in_byte);
            success &= CUDAcpyH2D((void *)cu_lps, (void *)h_lps, size_in_lbyte);

            if(success){
                gpuMemUploaded = true;
            }
            return success;
        }else{
            std::cout<<"memory not allocated, please allocate memory on gpu before copy data!"<<std::endl;
            return false;
        } 
    }

    /* malloc space and copy data to device */
    bool getReadyForGPU(){
        bool sucess = allocateDeviceSpace();
        sucess &= cmp2Device();
        return sucess;
    }

     void computeScatterRadianceAOGWithProcessRecordingAndDirectLightOPTGaussian(int iteration){
        if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);

        if(gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            size_t radiance_size_in_byte = num * sizeof(val3f);
            std::cout<<"num: "<< num <<" size_in_byte:"<<radiance_size_in_byte<<std::endl;
            val3f * cu_direct_radiance;
            CUDAmalloc((void **)&cu_direct_radiance, radiance_size_in_byte);
            computeDirectScatterAllOnGPUWithWeight(cu_sps, cu_lps, cu_neighbors, cu_dknn, num, maxK, cu_direct_radiance, pitch_n);
            CUDAcpyD2H((void *)cu_direct_radiance, m_result.blur_direct, radiance_size_in_byte);
            computeScatterAllOnGPURecordWithDirectOptandWeight(cu_sps, cu_neighbors, cu_knn, num, iteration, maxK, cu_direct_radiance, m_result, pitch_n);
            CUDAdelete(cu_direct_radiance);
        }
    }

    void computeScatterRadianceAOGWithDirectLightOPTGaussian(int iteration, bool jitter=false){
        if (1 != m_result.iter)
            allocateResultSpaceForNiteration(1);

        if(gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            size_t radiance_size_in_byte = num * sizeof(val3f);
            std::cout<<"num: "<< num <<" size_in_byte:"<<radiance_size_in_byte<<std::endl;
            val3f * cu_direct_radiance;
            CUDAmalloc((void **)&cu_direct_radiance, radiance_size_in_byte);
            computeDirectScatterAllOnGPUWithWeight(cu_sps, cu_lps, cu_neighbors, cu_dknn, num, maxK, cu_direct_radiance, pitch_n);
            CUDAcpyD2H((void *)cu_direct_radiance, m_result.blur_direct, radiance_size_in_byte);
            computeScatterAllOnGPUWithDirectOptandWeight(cu_sps, cu_neighbors, cu_knn,cu_dknn, num, iteration, maxK, dknn, cu_direct_radiance, m_result, pitch_n, jitter);
            CUDAdelete(cu_direct_radiance);
        }
    }

    void computeScatterRadianceAOGWithProcessRecordingAndDirectLightOPT(int iteration){
        if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);

        if(gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            size_t radiance_size_in_byte = num * sizeof(val3f);
            std::cout<<"num: "<< num <<" size_in_byte:"<<radiance_size_in_byte<<std::endl;
            val3f * cu_direct_radiance;
            CUDAmalloc((void **)&cu_direct_radiance, radiance_size_in_byte);
            computeDirectScatterAllOnGPU(cu_sps, cu_lps, cu_neighbors, cu_dknn, num, maxK, cu_direct_radiance, pitch_n);
            CUDAcpyD2H((void *)cu_direct_radiance, m_result.blur_direct, radiance_size_in_byte);
            computeScatterAllOnGPURecordWithDirectOpt(cu_sps, cu_neighbors, cu_knn, num, iteration, maxK, cu_direct_radiance, m_result, pitch_n);
            CUDAdelete(cu_direct_radiance);
        }
    }

    void computeScatterRadianceAOGWithDirectLightOPT(int iteration, bool jitter=false){
        if (1 != m_result.iter)
            allocateResultSpaceForNiteration(1);

        if(gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            size_t radiance_size_in_byte = num * sizeof(val3f);
            std::cout<<"num: "<< num <<" size_in_byte:"<<radiance_size_in_byte<<std::endl;
            val3f * cu_direct_radiance;
            CUDAmalloc((void **)&cu_direct_radiance, radiance_size_in_byte);
            computeDirectScatterAllOnGPU(cu_sps, cu_lps, cu_neighbors, cu_dknn, num, maxK, cu_direct_radiance, pitch_n);
            CUDAcpyD2H((void *)cu_direct_radiance, m_result.blur_direct, radiance_size_in_byte);
            computeScatterAllOnGPUWithDirectOpt(cu_sps, cu_neighbors, cu_knn,cu_dknn, num, iteration, maxK, dknn, cu_direct_radiance, m_result, pitch_n, jitter);
            CUDAdelete(cu_direct_radiance);
        }
    }

    void coteScatterRadianceAOGWithProcessRecording(int iteration){
        if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);

        if(gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            computeScatterAllOnGPURecord(cu_sps, cu_neighbors, cu_knn, num, iteration, maxK, m_result, pitch_n);
        }
    }

    void computeMISRadianceAOGWithProcessRecording(int iteration){
        if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);
        if(gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            computeMISAllOnGPURecord(cu_sps, cu_neighbors, cu_knn, num, iteration, maxK, m_result, pitch_n);
        }
    }

    void computeMISRadianceAOG(int iteration, bool jitter=false){
        if (1 != m_result.iter)
            allocateResultSpaceForNiteration(1);
        if(gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            computeMISAllOnGPU(cu_sps, cu_neighbors, cu_knn,cu_dknn, num, dknn, iteration, maxK, m_result, pitch_n, jitter);
        }
    }

    void computeScatterRadianceAOG(val3f *radiances, int iterations){
        if (gpuMemAllocated && gpuMemUploaded && neighborhoodcomputed){
            size_t size_in_byte = num * sizeof(val3f);
            std::cout<<"num: "<< num <<" size_in_byte:"<<size_in_byte<<std::endl;
            val3f *cu_radiance;
            CUDAmalloc((void **)&cu_radiance, size_in_byte);
            computeScatterAllOnGPU(cu_sps, cu_neighbors, cu_knn, num, iterations, maxK, cu_radiance, pitch_n);
            CUDAcpyD2H((void *)cu_radiance, (void *)radiances, size_in_byte);
            std::cout<<"complete copy radiance to host"<<std::endl;
            CUDAdelete(cu_radiance);
        }else{
            std::cout<<"data not ready, not computing!"<<std::endl;
        }
    }

    void ClusterScatter(int iteration){
        if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);

        std::cout<<"gpuMemAllocated: "<<gpuMemAllocated<<" gpuMemUploaded: "<<gpuMemUploaded<<" hasclusters: "<<hasclusters<<std::endl;

        if(gpuMemAllocated && gpuMemUploaded && hasclusters){
            computeClusterScatterAllOnGPURecord(cu_sps, numberOfClusters, cu_clusters, cu_clusters_offsets, num, iteration, m_result);
        }
    }

    void loadClusterScatter(int iteration){
        if(iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);
        
        if(gpuMemAllocated && gpuMemUploaded && hasclusters){
            ClusterScatterRecord(cu_sps, numberOfClusters, cu_clusters, cu_clusters_offsets, num, iteration, m_result);
        }
    }

    void ClusterScatter2(int iteration){
        if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);

        if(gpuMemAllocated && gpuMemUploaded && hasclusters){
            size_t *cu_element_offset;
            CUDAmalloc((void **)&cu_element_offset, sizeof(size_t) * num);
            size_t num_of_element = MatrixElementsNumber(cu_sps, cu_clusters_offsets, numberOfClusters, cu_element_offset,num);
            // std::cout<<"number of element "<<num_of_element<<std::endl;
            val3f *cu_matrix_elements;
            CUDAmalloc((void **)&cu_matrix_elements, sizeof(val3f) * num_of_element);
            std::cout<<"allocating "<<sizeof(val3f) * num_of_element / (float)(1024.0 * 1024.0)<<" mb "<<std::endl;
            precomputedMatrixElemtns(cu_sps, cu_clusters, cu_clusters_offsets, cu_element_offset, numberOfClusters, num, num_of_element, cu_matrix_elements);
            ClusterIterations(cu_sps, cu_clusters, cu_clusters_offsets, numberOfClusters, cu_element_offset, num, cu_matrix_elements, iteration, m_result);
            CUDAdelete(cu_matrix_elements);
            CUDAdelete(cu_element_offset);
        }
    }

    void ClusterScatterWithDirectOpt(int iteration){
         if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);

        if(gpuMemAllocated && gpuMemUploaded && hasclusters){
            val3f * cu_direct_radiance;
            CUDAmalloc((void **)&cu_direct_radiance, num * sizeof(val3f));
            ClusterDirect(cu_sps, cu_lps, cu_clusters, cu_clusters_offsets, num, numberOfClusters, cu_direct_radiance);

            size_t *cu_element_offset;
            CUDAmalloc((void **)&cu_element_offset, sizeof(size_t) * num);
            size_t num_of_element = MatrixElementsNumber(cu_sps, cu_clusters_offsets, numberOfClusters, cu_element_offset,num);
            // std::cout<<"number of element "<<num_of_element<<std::endl;
            val3f *cu_matrix_elements;
            CUDAmalloc((void **)&cu_matrix_elements, sizeof(val3f) * num_of_element);
            std::cout<<"allocating "<<sizeof(val3f) * num_of_element / (float)(1024.0 * 1024.0)<<" mb "<<std::endl;
            precomputedMatrixElemtns(cu_sps, cu_clusters, cu_clusters_offsets, cu_element_offset, numberOfClusters, num, num_of_element, cu_matrix_elements);
            ClusterIterations2(cu_sps, cu_clusters, cu_clusters_offsets, numberOfClusters, cu_element_offset, num, cu_matrix_elements, cu_direct_radiance, iteration, m_result);
            CUDAcpyD2H((void *)cu_direct_radiance, m_result.blur_direct, num * sizeof(val3f));

            CUDAdelete(cu_matrix_elements);
            CUDAdelete(cu_element_offset);
            CUDAdelete(cu_direct_radiance);
        }
    }

    void ClusterScatterWithDirectOptNR(int iteration){
         if (iteration != m_result.iter)
            allocateResultSpaceForNiteration(iteration);

        if(gpuMemAllocated && gpuMemUploaded && hasclusters){
            val3f * cu_direct_radiance;
            CUDAmalloc((void **)&cu_direct_radiance, num * sizeof(val3f));
            ClusterDirect(cu_sps, cu_lps, cu_clusters, cu_clusters_offsets, num, numberOfClusters, cu_direct_radiance);

            size_t *cu_element_offset;
            CUDAmalloc((void **)&cu_element_offset, sizeof(size_t) * num);
            size_t num_of_element = MatrixElementsNumber(cu_sps, cu_clusters_offsets, numberOfClusters, cu_element_offset,num);
            // std::cout<<"number of element "<<num_of_element<<std::endl;
            val3f *cu_matrix_elements;
            CUDAmalloc((void **)&cu_matrix_elements, sizeof(val3f) * num_of_element);
            std::cout<<"allocating "<<sizeof(val3f) * num_of_element / (float)(1024.0 * 1024.0)<<" mb "<<std::endl;
            precomputedMatrixElemtns(cu_sps, cu_clusters, cu_clusters_offsets, cu_element_offset, numberOfClusters, num, num_of_element, cu_matrix_elements);
            ClusterIterations3(cu_sps, cu_clusters, cu_clusters_offsets, numberOfClusters, cu_element_offset, num, cu_matrix_elements, cu_direct_radiance, iteration, m_result);
            CUDAcpyD2H((void *)cu_direct_radiance, m_result.blur_direct, num * sizeof(val3f));

            CUDAdelete(cu_matrix_elements);
            CUDAdelete(cu_element_offset);
            CUDAdelete(cu_direct_radiance);
        }
    }

    void loadClusters(int K){
        CUDAmalloc((void **)&cu_clusters_offsets, sizeof(int) * numberOfClusters);
        CUDAcpyH2D(cu_clusters_offsets, h_clusters_offset, sizeof(int) * numberOfClusters);

        CUDAmalloc((void **)&cu_clusters, sizeof(int) * numberOfPoints);
        CUDAcpyH2D(cu_clusters,h_clusters, sizeof(int) * numberOfPoints);
        std::cout<<" numberOfPoints = "<<numberOfPoints<<" num = "<<num<<std::endl;
        maxK = K;
        hasclusters = true;
    }

    void BuildClusters(int K){
        std::vector<int> v(num);
        // std::cout<<std::endl;
        std::generate(v.begin(), v.end(), [n=0]()mutable{return n++;});
        // for (int i = 0 ; i < 20 ; i++){
        //     std::cout<<v[i]<<" ";
        // }
        // std::random_device rd;
        // std::mt19937 g(rd());
        std::srand(1994);
        std::random_shuffle(v.begin(), v.end());
        // std::random_shuffle(v.begin(), v.end());
        // std::cout<<std::endl;
        // for (int i = 0 ; i < 20 ; i++){
        //     std::cout<<v[i]<<" ";
        // }
        // std::cout<<std::endl;
        maxK = K;

        float cellsize[3] = {(my_maxb[0] - my_minb[0]) / (float) my_dim[0], (my_maxb[1] - my_minb[1]) /  (float) my_dim[1], (my_maxb[2] - my_minb[2]) /  (float) my_dim[2]};
        
        hparam hp{
            {my_dim[0], my_dim[1], my_dim[2]}, 
            {cellsize[0], cellsize[1], cellsize[2]},
            {my_minb[0], my_minb[1], my_minb[2]},
            K,
            K
        };

        int numClusters = num / K + 1;
        int tablecells = numClusters;
        int *cu_np_points;

        // std::cout<<"num of total point: "<<num<<std::endl;
        // std::cout<<"num of cluster point: "<<numClusters<<std::endl;
        // std::cout<<"num of tablecells: "<<tablecells<<std::endl;

        CUDAmalloc((void **)&cu_clusters, sizeof(int) * num);
        CUDAmalloc((void **)&cu_np_points, sizeof(int) * numClusters);
        

        buildBatchClusters(cu_sps, num, tablecells, hp, numClusters, v.data(), cu_clusters, cu_np_points);

        // int *myclusters;
        // myclusters = (int*) malloc(sizeof(int) * numClusters);
        // CUDAcpyD2H(cu_np_points, myclusters, sizeof(int) * numClusters);
        // std::ofstream file2("/home/xd/Research/pathrenderer/scenes/kitchen/scene_clusters.bin", std::ios::out|std::ios::binary|std::ios::trunc);
        // if(file2.is_open()){
        //     file2.seekp(0);
        //     file2.write((char *)myclusters, sizeof(int) * numClusters);
        // }
        // file2.close();
        // delete[] myclusters;

        int *cu_num_small_clusters;
        CUDAmalloc((void **)&cu_num_small_clusters, sizeof(int) * numClusters);
        int new_num_clusters = countMyClusters(cu_sps, num, hp, numClusters, cu_clusters, cu_np_points, cu_num_small_clusters);

        // int *mynewclusters;
        // mynewclusters = (int*) malloc(sizeof(int) * numClusters);
        // CUDAcpyD2H(cu_num_small_clusters, mynewclusters, sizeof(int) * numClusters);
        // std::ofstream file3("/home/xd/Research/pathrenderer/scenes/kitchen/scene_new_clusters.bin", std::ios::out|std::ios::binary|std::ios::trunc);
        // if(file3.is_open()){
        //     file3.seekp(0);
        //     file3.write((char *)mynewclusters, sizeof(int) * numClusters);
        // }
        // file3.close();
        // delete[] mynewclusters;

        int *cu_num_point_new_cluster;
        CUDAmalloc((void **)&cu_num_point_new_cluster, sizeof(int) * new_num_clusters); 
        subClusters(cu_sps, num, hp, numClusters, new_num_clusters, cu_clusters, cu_np_points, cu_num_point_new_cluster, cu_num_small_clusters);

        // int *mynewclusters1;
        // mynewclusters1 = (int*) malloc(sizeof(int) * new_num_clusters);
        // CUDAcpyD2H(cu_num_point_new_cluster, mynewclusters1, sizeof(int) * new_num_clusters);
        // std::ofstream file3("/home/xd/Research/pathrenderer/scenes/kitchen/scene_new_sub_clusters.bin", std::ios::out|std::ios::binary|std::ios::trunc);
        // if(file3.is_open()){
        //     file3.seekp(0);
        //     file3.write((char *)mynewclusters1, sizeof(int) * new_num_clusters);
        // }
        // file3.close();
        // delete[] mynewclusters1;

        CUDAdelete(cu_num_small_clusters);
        CUDAmalloc((void **)&cu_num_small_clusters, sizeof(int) * new_num_clusters);
        int new_num_clusters2 = countMyClusters(cu_sps, num, hp, new_num_clusters, cu_clusters, cu_num_point_new_cluster, cu_num_small_clusters);


        int *cu_num_point_new_cluster2;
        CUDAmalloc((void **)&cu_num_point_new_cluster2, sizeof(int) * new_num_clusters2); 
        subClusters(cu_sps, num, hp, new_num_clusters, new_num_clusters2, cu_clusters, cu_num_point_new_cluster, cu_num_point_new_cluster2, cu_num_small_clusters);

        // int *mynewclusters2;
        // mynewclusters2 = (int*) malloc(sizeof(int) * new_num_clusters2);
        // CUDAcpyD2H(cu_num_point_new_cluster2, mynewclusters2, sizeof(int) * new_num_clusters2);
        // std::ofstream file3("/home/xd/Research/pathrenderer/scenes/veach-ajar/scene_new_sub_clusters_2.bin", std::ios::out|std::ios::binary|std::ios::trunc);
        // if(file3.is_open()){
        //     file3.seekp(0);
        //     file3.write((char *)mynewclusters2, sizeof(int) * new_num_clusters2);
        // }
        // file3.close();
        // delete[] mynewclusters2;

        FinalizeCluster(cu_sps, num, hp, new_num_clusters2, cu_clusters, cu_num_point_new_cluster2);
        // int *cu_num_in_cell;
        // CUDAmalloc((void **)&cu_num_in_cell, sizeof(int) * (numClusters));
        // FinalizeCluster(cu_sps, num, hp, numClusters, cu_clusters, cu_np_points, cu_num_in_cell);

        numberOfClusters =  new_num_clusters2;
        numberOfPoints = num;

        CUDAmalloc((void **)&cu_clusters_offsets, sizeof(int) * new_num_clusters2);
        CUDAmemD2D(cu_clusters_offsets, cu_num_point_new_cluster2, sizeof(int) * new_num_clusters2);
        
        // CUDAmalloc((void **)&cu_clusters_offsets, sizeof(int) * numberOfClusters);
        // CUDAmemD2D(cu_clusters_offsets, cu_num_in_cell, sizeof(int) * numberOfClusters);

        // int npp;
        // CUDAcpyD2H(cu_clusters_offsets + new_num_clusters2 - 1, &npp, sizeof(int));
        // std::cout<<"            MY CLUSTER POINTS          "<<npp<<"  V.S. "<<num<<std::endl;

     

        // int *mynewclusterscc;
        // mynewclusterscc = (int*) malloc(sizeof(int) * num);
        // CUDAcpyD2H(cu_clusters, mynewclusterscc, sizeof(int) * num);
        // std::ofstream file4("/home/xd/Research/pathrenderer/scenes/veach-ajar/scene_new_sub_clusters_idx.bin", std::ios::out|std::ios::binary|std::ios::trunc);
        // if(file4.is_open()){
        //     file4.seekp(0);
        //     file4.write((char *)mynewclusterscc, sizeof(int) * num);
        // }
        // file4.close();
        // delete[] mynewclusterscc;
        
        hasclusters = true;
        CUDAdelete(cu_np_points);
        // CUDAdelete(cu_num_small_clusters);
        // CUDAdelete(cu_num_point_new_cluster);
        // CUDAdelete(cu_num_point_new_cluster2);
    }

    void BuildKNN(int K, int dK){
        knn = K;
        dknn = dK;
        size_t size_of_int_in_byte = num * sizeof(int);
        /* compute the size of new memory */
        maxK = std::max(K, dK);
        size_t ind_size_in_byte = num * maxK * sizeof(int);
        std::cout<<"\n Warn!! You are allocating"<<(float)ind_size_in_byte / (1024 * 1024 * 1024)<<"GB on GPU!"<<std::endl;
        printf("\n maxK=%d, k=%d, dk=%d, num of point = %d", maxK, knn, dknn, num);

        /* if the memory are previously allocated, remove them */
        /*allocate the memory on gpu and cpu*/
        CUDAmallocPitch((void **)&cu_neighbors, &pitch_n, maxK * sizeof(int), num);
        CUDAmalloc((void **)&cu_knn, size_of_int_in_byte);
        CUDAmalloc((void **)&cu_dknn, size_of_int_in_byte);
        float cellsize[3] = {(my_maxb[0] - my_minb[0]) / my_dim[0], (my_maxb[1] - my_minb[1]) / my_dim[1], (my_maxb[2] - my_minb[2]) / my_dim[2]};
        
        hparam hp{
            {my_dim[0], my_dim[1], my_dim[2]}, 
            {cellsize[0], cellsize[1], cellsize[2]},
            {my_minb[0], my_minb[1], my_minb[2]},
            K,
            dK
        };

        buildHashGridAndFindKNN(cu_sps, hp, cu_neighbors, num, cu_knn, cu_dknn, pitch_n);
        neighborhoodcomputed = true;
    }

    SPoints(/* args */){
        num=0;
        m_grid = HashGrid();
    }

    ~SPoints()
    {   
        std::cout<<"deconstruct SPoints"<<std::endl;
        if(num != 0){
            free(h_sps);
            freeDeviceSpace();
            freelightpoints();
            freeClustersSpace();
            freeCudaNeighbors();
        }
        std::cout<<"deconstruct SPoints 2"<<std::endl;
        if(m_result.iter > 0){
            freeResultSpace();
        }
        std::cout<<"deconstruct SPoints 3"<<std::endl;
        if(hashostclusters){
            free(h_clusters);
            free(h_clusters_offset);
            hashostclusters = false;
        }
    }

    SPoints(int N) {
        allocateHostSpace(N);
    }
};

