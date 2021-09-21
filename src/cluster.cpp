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

void writeIndirectLight(string filename, Vector2i size, const PathGraph *pg){
    Bitmap iter_image(size);
    iter_image.setConstant(Color3f(0.0f));
    Bitmap iter_image_c(size);
    iter_image_c.setConstant(Color3f(0.0f));
    Bitmap iter_image_b(size);
    iter_image_b.setConstant(Color3f(0.0f));
    val3f *ptr = pg->m_sps.m_result.mc_results[pg->m_sps.m_result.iter-1];
    val3f *bptr = pg->m_sps.m_result.blur_results[pg->m_sps.m_result.iter-1];
    for (int j = 0 ; j < pg->m_pathCount ; j++){
        int r = pg->m_cpl[j].xIdx;
        int c = pg->m_cpl[j].yIdx;
        if(pg->m_cpl[j].numOfPathPoints > 0){
            int pid = pg->m_cpl[j].firstPathPointIdx;
            float red = (ptr+pid)->value[0];
            float green = (ptr+pid)->value[1];
            float blue = (ptr+pid)->value[2];
            // if ( r == 1185 && c == 36 ){
            //     std::cout<<"Index = "<<pid<<" x = "<<r<<" y = "<<c<<" number of point: "<<pg->m_cpl[j].numOfPathPoints<<std::endl;
            // }
                
            iter_image(c, r) = Color3f(red, green, blue);
            
            red = pg->m_sps.h_sps[pid].eLi[0] - pg->m_sps.h_sps[pid].eLd[0];
            green = pg->m_sps.h_sps[pid].eLi[1] - pg->m_sps.h_sps[pid].eLd[1];
            blue = pg->m_sps.h_sps[pid].eLi[2] - pg->m_sps.h_sps[pid].eLd[2];
            iter_image_c(c,r) = Color3f(red, green, blue);

            red = (bptr+pid)->value[0];
            green = (bptr+pid)->value[1];
            blue = (bptr+pid)->value[2];
            iter_image_b(c,r) = Color3f(red, green, blue);
        }
    }
    iter_image.saveEXR(filename);
    iter_image_c.saveEXR(filename + "_pt");
    iter_image_b.saveEXR(filename + "_blur");
}

void writeFullinit(string filename, Vector2i size, const PathGraph *pg){
    Bitmap iter_image(size);
    iter_image.setConstant(Color3f(0.0f));
    val3f *ptr = pg->m_sps.m_result.mc_results[pg->m_sps.m_result.iter-1];
    for (int j = 0 ; j < pg->m_pathCount ; j++){
        int r = pg->m_cpl[j].xIdx;
        int c = pg->m_cpl[j].yIdx;
        if(pg->m_cpl[j].numOfPathPoints > 0){
            int pid = pg->m_cpl[j].firstPathPointIdx;
            float red = pg->m_sps.h_sps[pid].eLd[0];
            float green = pg->m_sps.h_sps[pid].eLd[1];
            float blue = pg->m_sps.h_sps[pid].eLd[2];
            iter_image(c, r) = Color3f(red, green, blue);
        }else{
            iter_image(c, r) = Color3f(pg->m_cpl[j].em[0], pg->m_cpl[j].em[1], pg->m_cpl[j].em[2]);
        }
    }
    iter_image.saveEXR(filename);
}

void writeFullLight(string filename, Vector2i size, const PathGraph *pg){
    Bitmap iter_image(size);
    iter_image.setConstant(Color3f(0.0f));
    val3f *ptr = pg->m_sps.m_result.mc_results[pg->m_sps.m_result.iter-1];
    for (int j = 0 ; j < pg->m_pathCount ; j++){
        int r = pg->m_cpl[j].xIdx;
        int c = pg->m_cpl[j].yIdx;
        
        if(pg->m_cpl[j].numOfPathPoints > 0){
            int pid = pg->m_cpl[j].firstPathPointIdx;
            float red = pg->m_sps.h_sps[pid].eLd[0] + (ptr+pid)->value[0];
            float green = pg->m_sps.h_sps[pid].eLd[1] + (ptr+pid)->value[1];
            float blue = pg->m_sps.h_sps[pid].eLd[2] + (ptr+pid)->value[2];
            iter_image(c, r) = Color3f(red, green, blue);
        }else{
            iter_image(c, r) = Color3f(pg->m_cpl[j].em[0], pg->m_cpl[j].em[1], pg->m_cpl[j].em[2]);
        }
    }
    iter_image.saveEXR(filename);
}

void writeDirectLight(string filename, val3f *direct_ptr, Vector2i size, const PathGraph *pg){
    Bitmap iter_image(size);
    iter_image.setConstant(Color3f(0.0f));
    Bitmap iter_image_o(size);
    iter_image_o.setConstant(Color3f(0.0f));
    for (int j = 0 ; j < pg->m_pathCount ; j++){
        if(pg->m_cpl[j].numOfPathPoints > 0){
            int pid = pg->m_cpl[j].firstPathPointIdx;
            int r = pg->m_cpl[j].xIdx;
            int c = pg->m_cpl[j].yIdx;
            float red = (direct_ptr+pid)->value[0];
            float green = (direct_ptr+pid)->value[1];
            float blue = (direct_ptr+pid)->value[2];
            iter_image(c, r) = Color3f(red, green, blue);

            red = pg->m_sps.h_sps[pid].eLd[0];
            green = pg->m_sps.h_sps[pid].eLd[1];
            blue = pg->m_sps.h_sps[pid].eLd[2];
            iter_image_o(c, r) = Color3f(red, green, blue);
        }
    }
    iter_image.saveEXR(filename);
    iter_image_o.saveEXR(filename+"_o");
}

void writeProgressImage(string filename, val3f *blur_ptr, val3f *mc_ptr, int iter, Vector2i size, const PathGraph *pg){
        Bitmap iter_image(size);
        Bitmap iter_image_mc(size);
        iter_image.setConstant(Color3f(0.0f));
        iter_image_mc.setConstant(Color3f(0.0f));
        for (int j = 0 ; j < pg->m_pathCount ; j++){
            if(pg->m_cpl[j].numOfPathPoints > 0){
                int pid = pg->m_cpl[j].firstPathPointIdx;
                int r = pg->m_cpl[j].xIdx;
                int c = pg->m_cpl[j].yIdx;
                float red = (blur_ptr+pid)->value[0];
                float green = (blur_ptr+pid)->value[1];
                float blue = (blur_ptr+pid)->value[2];
                iter_image(c, r) = Color3f(red, green, blue);

                red = (mc_ptr + pid)->value[0];
                green = (mc_ptr + pid)->value[1];
                blue = (mc_ptr + pid)->value[2];
                iter_image_mc(c, r) = Color3f(red, green, blue);
            }
        }
        
        string file = filename + "_iter_" + to_string(iter);
        iter_image.saveEXR(file);
        string file2 = filename + "_mc_iter_" + to_string(iter);
        iter_image_mc.saveEXR(file2);
}

int main(int argc, char **argv){
    cout<<"argc:"<<argc<<"argv: "<<argv[1];
    PathGraph pg;
    int m_k = 16;
    int m_iteration = 1;
    string filename = "";
    string m_mode = "opt";
    string foldername = "";

    bool loaded = false;
    if(argc > 1){
       foldername = "/home/xd/Research/pathrenderer/scenes/" + string(argv[1]);
        // foldername = "/media/xd/Data/data/outputs/" + string(argv[1]);
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
        }else if (str.compare(string("-m")) == 0){
            m_mode = string(argv[i+1]);
            cout<<"mode = "<<m_mode<<endl;
        }
    }

    if (loaded){
        pg.m_sps.getReadyForGPU();
        cout<<"Info about total shading points: "<<pg.m_sps.num<<endl;
        cout<<"Info about total paths: "<<pg.m_pathCount<<endl;
        clock_t start, end;
        start = clock();
        if(m_mode.compare("knn") == 0){
            pg.m_sps.BuildKNN(m_k, m_k);
        }else if(m_mode.compare("l") == 0){
            pg.loadNeighbors(foldername);
            pg.m_sps.loadClusters(m_k);
        }else{
            pg.m_sps.BuildClusters(m_k);
        }
        
        end = clock();
        double time_exe = double(end - start) / double(CLOCKS_PER_SEC);
        cout<<"Time taken by building cluster is: "<<fixed<<time_exe<<setprecision(5);
         cout<<" sec "<<endl;
        start = clock();
        if (m_mode.compare("opt") == 0){
            pg.m_sps.ClusterScatterWithDirectOptNR(m_iteration);
            // pg.m_sps.ClusterScatterWithDirectOpt(m_iteration);
        }else if(m_mode.compare("n") == 0){
            pg.m_sps.ClusterScatter2(m_iteration);
        }else if(m_mode.compare("t") == 0){
            pg.m_sps.ClusterScatter(m_iteration);
        }else if(m_mode.compare("l") == 0){
            pg.m_sps.loadClusterScatter(m_iteration);
        }else if(m_mode.compare("knn") == 0){
            pg.m_sps.computeMISRadianceAOGWithProcessRecording(m_iteration);
        }
        
        end = clock();
        time_exe = double(end - start) / double(CLOCKS_PER_SEC);
        cout<<"Time taken by iteration is: "<<fixed<<time_exe<<setprecision(5);
        cout<<" sec "<<endl;

        int width = pg.m_xresolution;
        int height = pg.m_yresolution;

        string dfile = filename + "_k-" + to_string(m_k) + "_direct";
        writeDirectLight(dfile, pg.m_sps.m_result.blur_direct, Vector2i(width, height), &pg);
        writeFullinit(filename + "_Le_init", Vector2i(width, height), &pg);
        string tfile = filename + "_k-" + to_string(m_k) + "_full";
        writeFullLight(tfile, Vector2i(width, height), &pg);
        tfile = filename + "_k-" + to_string(m_k) + "_indirect";
        writeIndirectLight(tfile, Vector2i(width, height), &pg);
        // for (int i = 0 ; i < pg.m_sps.m_result.iter ; i++){
        //     cout<<"writing iteration "<<i<<endl;
        //     string bfile = filename + "_k-" + to_string(m_k) + "_indirect_" + to_string(i);
        //     writeBinaryFile(bfile, sizeof(val3f) * pg.m_sps.num, (char *) pg.m_sps.m_result.blur_results[i]);

        //     string file = filename + "_k-" + to_string(m_k);
        //     writeProgressImage(file, pg.m_sps.m_result.blur_results[i],pg.m_sps.m_result.mc_results[i],i,Vector2i(width, height), &pg);
        // }
    }

    return 0;
}