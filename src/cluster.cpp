#include <iostream>
#include <string>
#include "nori/pathgraph.h"
#include <bits/stdc++.h>
#include <nori/bitmap.h>

using namespace std;
using namespace nori;

bool writeBinaryFile(string filename, size_t datasize ,char *data){
    std::ofstream file2(filename + ".bin", std::ios::out|std::ios::binary|std::ios::trunc);
    if(file2.is_open()){
        file2.seekp(0);
        file2.write((char *)data, datasize);
    } else {
        cout<<"File load failed"<<endl;
        return false;
    }
    file2.close();
    return true;
}

int main(int argc, char **argv){
    cout<<"argc:"<<argc<<"argv: "<<argv[1];
    PathGraph pg;
    int m_k = 16;
    int m_iteration = 1;
    string filename = "";

    bool loaded = false;
    if(argc > 1){
       string foldername = "/home/xd/Research/pathrenderer/scenes/" + string(argv[1]);
       filename = filename + foldername;
       loaded = pg.loadShadingPoints(foldername);
       loaded &= pg.loadLightPoints(foldername);
       loaded &= pg.loadPaths(foldername);
       loaded &= pg.loadAABB(foldername);
       pg.computeDimensions(pg.m_aabb, pg.m_spCount);
    } else {
        cout<<"Error: No filename was given!";
    }

    for(int i = 2 ; i < argc ; i += 2){
        cout<<endl;
        cout<<argv[i]<<endl;
        string str(argv[i]);
        if (str.compare(string("-k")) == 0){
            m_k = stoi(argv[i+1]);
            cout<<"m_k = "<<m_k<<endl;
        }else if (str.compare(string("-i")) == 0){
            m_iteration = stoi(argv[i+1]);
            cout<<"iteration = "<<m_iteration<<endl;
        }
    }

    if (loaded){
        pg.m_sps.getReadyForGPU();
        cout<<"Info about total shading points: "<<pg.m_sps.num<<endl;
        cout<<"Info about total paths: "<<pg.m_pathCount<<endl;
        clock_t start, end;
        start = clock();
        pg.m_sps.BuildClusters(m_k);
        end = clock();
        double time_exe = double(end - start) / double(CLOCKS_PER_SEC);
        cout<<"Time taken by building knn is: "<<fixed<<time_exe<<setprecision(5);
        cout<<" sec "<<endl;
        start = clock();
        pg.m_sps.ClusterScatter2(m_iteration);
        end = clock();
        time_exe = double(end - start) / double(CLOCKS_PER_SEC);
        cout<<"Time taken by iteration is: "<<fixed<<time_exe<<setprecision(5);
        cout<<" sec "<<endl;

        int width = pg.m_xresolution;
        int height = pg.m_yresolution;

        for (int i = 0 ; i < pg.m_sps.m_result.iter ; i++){
            
            string bfile = filename + "_indirect_" + to_string(i);
            writeBinaryFile(bfile, sizeof(val3f) * pg.m_sps.num, (char *) pg.m_sps.m_result.blur_results[i]);

            Bitmap iter_image(Vector2i(width, height));
            Bitmap iter_image_mc(Vector2i(width, height));
            iter_image.setConstant(Color3f(0.0f));
            iter_image_mc.setConstant(Color3f(0.0f));
            for (int j = 0 ; j < pg.m_pathCount ; j++){
                if(pg.m_cpl[j].numOfPathPoints > 0){
                    int pid = pg.m_cpl[j].firstPathPointIdx;
                    int r = pg.m_cpl[j].xIdx;
                    int c = pg.m_cpl[j].yIdx;
                    float red = (pg.m_sps.m_result.blur_results[i] + pid)->value[0];
                    float green = (pg.m_sps.m_result.blur_results[i] + pid)->value[1];
                    float blue = (pg.m_sps.m_result.blur_results[i] + pid)->value[2];
                    iter_image(c, r) = Color3f(red, green, blue);

                    red = (pg.m_sps.m_result.mc_results[i] + pid)->value[0];
                    green = (pg.m_sps.m_result.mc_results[i] + pid)->value[1];
                    blue = (pg.m_sps.m_result.mc_results[i] + pid)->value[2];
                    iter_image_mc(c, r) = Color3f(red, green, blue);
                }
            }
            
            string file = filename + "_k-" + to_string(m_k) + "_iter_" + to_string(i) + ".exr";
            iter_image.saveEXR(file);
            string file2 = filename + "_k-" + to_string(m_k) + "_mc_iter_" + to_string(i) + ".exr";
            iter_image_mc.saveEXR(file2);
        }

    }

    return 0;
}