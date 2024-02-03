#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80GB
##SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --job-name='SmallTown_r128s6'
##SBATCH -p nvidia
##SBATCH --reservation=gpu
##SBATCH -q cair

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

source ~/.bashrc
conda activate torch2.0_cuda11.8

export CUDA_PATH=/share/apps/NYUAD/cuda/11.8.0

min_res=4
alpha=10
base_noise_amp=0.1

#data='Acropolis_r128s6'
#data='StoneStairs_r128s7'
#data='Rock_r128s7'
#data='Wall_r128s6'
#data='Vase_r128s7'
#data='Cheese_r128s6'
#data='Cactus_r128s6'
#data='Tree_r128s6'
#data='Canyon_r128s6'
#data='NaturalArch_r128s7'
#data='Castle_r128s6'
data='SmallTown_r128s6'

aug_types=('cropping' 'flipping' 'rotation' 'cropping,flipping,rotation')

for aug_type in "${aug_types[@]}"; do
    tag=${data}'_v'${min_res}'a'${alpha}'_n'${base_noise_amp}'_'${aug_type}
    # If you want to visualize the generated meshes, uncomment '--no_vis'
    python vis_export.py -s checkpoints/${tag}/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
done

#python vis_export.py -s checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Acropolis_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Acropolis_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Acropolis_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Cactus_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cactus_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cactus_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cactus_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cactus_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cactus_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cactus_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Canyon_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Canyon_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Canyon_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Canyon_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Canyon_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Canyon_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Canyon_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Castle_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Castle_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Castle_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Castle_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Castle_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Castle_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Castle_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Cheese_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cheese_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cheese_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cheese_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cheese_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cheese_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Cheese_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/NaturalArch_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/NaturalArch_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/NaturalArch_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Rock_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Rock_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Rock_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Rock_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Rock_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Rock_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Rock_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/SmallTown_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/SmallTown_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/SmallTown_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/StoneStairs_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/StoneStairs_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/StoneStairs_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Tree_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Tree_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Tree_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Tree_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Tree_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Tree_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Tree_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Vase_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Vase_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Vase_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Vase_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Vase_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Vase_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Vase_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis

#python vis_export.py -s checkpoints/Wall_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Wall_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
python vis_export.py -s checkpoints/Wall_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Wall_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Wall_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Wall_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis
#python vis_export.py -s checkpoints/Wall_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0 --smooth 3 --export obj --cleanup  --no_vis