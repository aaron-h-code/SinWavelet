#!/bin/bash

source ~/.bashrc
#conda activate torch2.0_cuda11.8
conda activate sinwavelet

export CUDA_PATH=/share/apps/NYUAD/cuda/11.8.0

# Arguments
# -s: path to the mesh file
# --res finest voxel resolution
# --n_scales: number of scales
# -o output path and filename
python voxelization/voxelize.py -s data/single_shape_mesh/Acropolis.obj --res 256 --n_scales 5 -o data/Acropolis_r256s5.h5

