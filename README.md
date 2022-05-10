# Project of Uncertain Graph System
Toward Enabling efficient uncertain graph processing on multi-accelerator systems.

- BPGraph: Path-Sampling Centric Uncertain Graph Processing

# BPGraph (version Beta)

BPGraph is released under the MIT license.

If you use BPGraph in your research, please cite our paper:

- ICS'22, BPGraph

```
@inproceedings{cusz2020,
    title = {Bring Orders into Uncertainty: Enabling Efficient Uncertain Graph Processing via Novel Path Sampling on Multi-Accelerator Systems},
    author = {Zhang, Heng and Li, Lingda and Liu, Hang and Zhuang, Donglin and Liu, Rui and Huan, Chengying and Song, Shuang and Tao, Dingwen and Liu, Yongchao and He, Charles and Wu, Yanjun and Song, Shuaiwen Leon},
    year = {2022},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    doi = {10.1145/3524059.3532379},
    booktitle = {2022 International Conference on Supercomputing},
    location = {Virtual Event, USA},
    series = {ICS '22}
}
```

# Introduction

BPGraph is based on a novel runtime path sampling method, which is able to identify and eliminate unnecessary edge sampling via incremental path identification and filtering, resulting in significant reduction in computation and data movement. BPGraph is a general uncertain graph processing framework for multi-GPU systems, and provides general support for users to design and optimize a wide-range of uncertain graph algorithms and applications without concerning about the underlying complexity. For more details, see [BPGraph's Overview]().

BPGraph New Features:

- **Path Sampling Methodology** 
- **Path-Sampling Centric Programming APIs**
- **High-Performance Supporting on GPUs**

# Supporting Applications
BPGraph supports the following applications in the implementation, and other algorithms are under developing.

- Source-Target Path Query
- K-Nearest Neighbors
- Any-Pair Shortest Path Search

# Quick Start Guide

Before building BPGraph, we recommend our software environment configuration of *CMake 3.12*, *GCC 5.4* and *CUDA 11 are required*. The later version of the software dependencies are not tested in this time. Meanwhile, the third party code contained in our repository includes the following external dependencies: NCCL, BOOST, and other optional repository. For complete build guide, see [Building BPGraph]().

```
git clone --recursive https://github.com/bpgraph/bpgraph.git
cd bpgraph
mkdir build && cd build
cmake .. && make -j$(nproc)
bin/bpgraph --edgelist ../toy.mtx (--st ../st_toy.txt --app ...)
```

# Getting Started with BPGraph

Fast start to run a simple example of *source-to-target query* in BPGraph system.

```
bin/bpgraph --edgelist ../toy.mtx --st ../st_toy.txt
```


# Preprocessing

The dataset for this artifact contains all graphs listed the paper, converted to the edgelist matrix format (.mtx or .txt ﬁles). Also, we separately give the scripts of small and large datasets for automatically downloaded from open source repositories. Each graph is located in a separate subdirectory, with three additional ﬁles:

- **{graphfile}_mtx.txt** Raw uncertain graph file.
- **{graphfile}.metadata** Description file contains the basic information (\#vertex, \#edge, \#diameter) for uncertain graphs. 
- **{st-graphfile}.txt** Source-target query datasets containing the generated source-target query pairs.
- **Monte-Carlo.txt** Result datasets containing the final execution time, memory consumption and the reliability data. 


# Execution

Execute uncetain graph query with the following command. ``--edgelist`` sets the input graph file, ``-st`` sets the source-to-target node pair list file, ``-app`` sets the application and ``-d`` sets the number of GPU devices.

```
bin/bpgraph --edgelist ../toy.mtx --st ../st_toy.txt -d 2 -app st
```

Execute KNN with the following command.

```
bin/bpgraph --edgelist ../friendster.mtx -d 2 -app knn -sid 0

```

Execute ASAP with the follwoing command. 

```
bin/bpgraph --edgelist ../twitter.mtx --st ../st_toy.txt -d 2 -app asap
```


# Evaluation

The source files are under ``scripts`` folder. The parameters have been configured in the source files. You can rewrite the default setting (e.g., the number of devices, the sampling method and the distance metrics) in the source files or through parameters with the instructions in the files. Here, we demonstrate the input of graph data.

## Experiment Datasets

You can download the graphs used in our paper by following the instructions.

## Experimental Scritpts

The source files are under ``scripts`` folder.
