import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
import json
import random
import subprocess
from subprocess import check_output

# rootdir = "/media/xd/Data/data/0715/outputs/"
rootdir = "/media/xd/Data/data/0715/nonclamp/"
outdir = "/media/xd/Data/data/0715/nonclamp/"
# rootdir = "/home/xd/Research/pathrenderer/outputs/"
# outdir = "/home/xd/Research/pathrenderer/outputs/"

sceneList = {
    # 'veach-ajar',
    "staircase",
    # 'staircase2',
    'office',
    # 'living-room-3',
    # 'dining-room-2',
    # 'classroom',
    'dining-room',
    # 'living-room',
    'living-room-2',
    'kitchen'
}

oldmergeDict1 = {
    # "veach-ajar":14,
    # "office":16,
    "living-room-3":15,
    "dining-room-2":19,
    "staircase":16,
    "staircase2":16,
    "kitchen":16,
    # "living-room":16,
    "dining-room":16,
    # "classroom":17,
    # "living-room-2":15
}

mergeDict1 = {
    "veach-ajar":24,
    "staircase":23,
    "office":19,
    "kitchen":23,
    "kitchen":23,
    "dining-room":24,
    "classroom":23,
    "living-room-2":18,
    "dining-room-2":21,
    "living-room-3":19,
    "caustics":65
}

mergeDict2 = {
    # "veach-ajar":30,
    "office":21,
    "living-room-3":20,
    "dining-room-2":25,
    "staircase":25,
    "staircase2":25,
    "kitchen":25,
    "living-room":28,
    "dining-room":26,
    "classroom":27,
    "living-room-2":24
}

def runPathGraph(name):
    cmd=[]
    cmd.append("~/Research/vis/build/pg {} -k 16 -i 3 -m opt".format(name))
    print(cmd)
    subprocess.call(cmd, shell=True)

def mergeGraph(name, output):
    cmd=[]
    cmd.append("/home/xd/Projects/utils/tungsten/build/release/hdrmanip --merge {} --output {}".format(name, output))
    print(cmd)
    subprocess.call(cmd, shell=True)

def runAllPathGraph():
    for scene in sceneList:
        for i in range(0, 22):
            fn = scene + '/' + "A_{:02d}".format(i)
            runPathGraph(fn)

def MergePathGraphs():
    for scene in sceneList:
        fn = rootdir + scene + '/' + "A_*_k-16_indirect.exr"
        out = rootdir + scene + '/' + "A_k-16_indirect_222spp.exr"
        mergeGraph(fn, out)


def MergePathNGraphs(option):
    for scene in sceneList:
        mer = " "
        if option == 1:
            for i in range(mergeDict1[scene]):
                mer += " " + rootdir+scene+'/'+"A_{:02d}_k-16_indirect.exr".format(i)
            mer + " "
            out = outdir + scene + '/' + "A_k-16_indirect_{}spp.exr".format(mergeDict1[scene])
        else:
            for i in range(mergeDict2[scene]):
                mer += " " + rootdir+scene+'/'+"A_{:02d}_k-16_indirect.exr".format(i)
            mer + " "
            out = outdir + scene + '/' + "A_k-16_indirect_wo_{}spp.exr".format(mergeDict2[scene])
        mergeGraph(mer, out)

def MergePathGraphs(scene):
    mmer = " "
    for j in range(1, 200):
        mer = " "
        for i in range(100):
            iidx = j*100 + i
            mer += " " + rootdir + scene + "_clamp/" + "scene_{:05d}.exr".format(iidx)
        mer += ' '
        out = outdir + scene + '_clamp/' + 'merge_{}_{:03d}.exr'.format(scene, j)
        mergeGraph(mer, out)
        mmer += " " + outdir + scene + "_clamp/" + "merge_{}_{:03d}.exr".format(scene, j)
    mmer += ' '
    oout = outdir + scene + '_clamp/' + "{}_final.exr".format(scene)
    mergeGraph(mmer, oout)


numList = [1, 4, 16, 32, 64]
def MergePathNgGraphs():
    scene = 'office'
    for n in numList:
        mer = " "
        for i in range(n):
            mer += " " + rootdir+scene+'/'+"A_{:02d}_k-16_indirect.exr".format(i)
        mer + " "
        out = outdir + scene + '/' + "A_k-16_indirect_{}spp.exr".format(n)
        mergeGraph(mer, out)

def MergeEveryKGraphs(k, o):
    scene = 'kitchen'
    offset = o
    for y in range(10):
        mer = " "
        for i in range(k):
            mer += " " + rootdir + scene + '/' + "A_{:02d}_k-16_indirect.exr".format(i+offset)
        mer + " "
        out = outdir + scene + '/' + "A_k-16_indirect_{}spp_{:03d}.exr".format(k, y)
        mergeGraph(mer, out)
        offset += k



def computeRMSE(A, B):
    cmd = []
    cmd.append("/home/xd/Projects/utils/tungsten/build/release/hdrmanip --rmse {} {}".format(A, B))
    # print(cmd)
    # r = check_output(cmd, shell=True)
    # p = subprocess.Popen(cmd)
    r = subprocess.call(cmd, shell=True)
    return r

def computeRMSEReturn(A, B):
    cmd = []
    cmd.append("/home/xd/Projects/utils/tungsten/build/release/hdrmanip --rmse {} {}".format(A, B))
    r = check_output(cmd, shell=True)
    return r    

refDict = {
    "veach-ajar":107,
    "office":37,
    "living-room-3":34,
    "dining-room-2":45,
    "staircase":50,
    "staircase2":50,
    "kitchen":60,
    "living-room":65,
    "dining-room":66,
    "classroom":61,
    "living-room-2":58,
    "caustics":110,
}

def toPNG(filename):
    cmd=[]
    cmd.append("/home/xd/Projects/utils/tungsten/build/release/hdrmanip -f png -e 2.0 {}".format(filename))
    subprocess.call(cmd, shell=True)

def plotRMSE():
    for scene in sceneList:
        print("=====================")
        print(scene)
        print("path tracing: ")
        B1 = rootdir + scene + "/" + "scene_{:02d}spp.exr".format(refDict[scene])
        A = rootdir + scene + "/" + "reference.exr"
        # A = rootdir + scene + "/" + "scene_indirect_ref_00.exr"
        rpt = computeRMSE(A, B1)

        print("path graph: ")
        B2 = rootdir + scene + '/' + "A_k-16_indirect_{}spp.exr".format(mergeDict1[scene])
        rour1 = computeRMSE(A, B2)

        print("bdpt: ")
        B3 = rootdir + scene + '/' + "bdpt_30sec.exr"
        bdpt = computeRMSE(A, B3)

        print("path-guiding: ")
        B4 = rootdir + scene + '/' + "path_guiding_30sec.exr"
        bdpt = computeRMSE(A, B4)
        # print(scene, float(rpt) / float(rour1))

        # print("idea: ")

        # B3 = rootdir + scene + '/' + "A_k-16_indirect_wo_{}spp.exr".format(mergeDict2[scene])
        # rour2 = computeRMSE(A, B3)

        # print(scene, float(rpt) / float(rour2))

def plotRMSERatio():
    for scene in sceneList:
        print("=====================")
        print(scene)
        print("ref: ")
        B1 = rootdir + scene + "/" + "scene_{:02d}spp.exr".format(refDict[scene])
        A = rootdir + scene + "/" + "reference.exr"
        # A = rootdir + scene + "/" + "scene_indirect_ref_00.exr"
        rpt = computeRMSEReturn(A, B1)

        print("path graph: ")
        B2 = rootdir + scene + '/' + "A_k-16_indirect_{}spp.exr".format(mergeDict1[scene])
        rour1 = computeRMSEReturn(A, B2)

        print(scene, float(rpt) / float(rour1))

        # print("idea: ")

        # B3 = rootdir + scene + '/' + "A_k-16_indirect_wo_{}spp.exr".format(mergeDict2[scene])
        # rour2 = computeRMSEReturn(A, B3)

        # print(scene, float(rpt) / float(rour2))        

def topngs():
    for scene in sceneList:
        # print("ref: ")
        B1 = rootdir + scene + "/" + "scene_{:02d}spp.exr".format(refDict[scene])
        # A = rootdir + scene + "/" + "scene_4096spp.exr"
        A = rootdir + scene + "/" + "reference.exr"
        B2 = rootdir + scene + '/' + "A_k-16_indirect_{}spp.exr".format(mergeDict1[scene])
        B3 = rootdir + scene + '/' + "bdpt_30sec.exr"
        B4 = rootdir + scene + '/' + "path_guiding_30sec.exr"

        toPNG(B1)
        toPNG(A)
        toPNG(B2)
        toPNG(B3)
        toPNG(B4)
        # toPNG(B5)


if __name__ == "__main__":
    # runAllPathGraph()
    # MergePathNGraphs(1)
    # MergePathNGraphs(2)
    # MergePathNgGraphs()
    # MergeEveryKGraphs(11, 200)
    # plotRMSE()
    # plotRMSERatio()
    # topngs()
    # MergePathGraphs("living-room")
    # MergePathGraphs("kitchen")
    MergePathGraphs("dining-room")
    # MergePathGraphs("living-room")