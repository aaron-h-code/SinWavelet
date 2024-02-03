#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
##SBATCH --gres=gpu:1
#SBATCH --gres=gpu:a100:1
##SBATCH --gres=gpu:rtx8000:2
#SBATCH --mem=80GB
##SBATCH --exclusive
#SBATCH --time=96:00:00
#SBATCH --job-name='SmallTown_r128s6'
#SBATCH -p nvidia
##SBATCH --reservation=gpu
##SBATCH -q cair

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

source ~/.bashrc
conda activate torch2.0_cuda11.8

min_res=4
alpha=10
base_noise_amp=0.1

gpu_id=0

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

ref='../data/'${data}'.h5'

#ref='/scratch/hh1811/projects/CDDPM-v10/data/Acropolis_r128s6.h5'

for aug_type in "${aug_types[@]}"; do
    fake='../checkpoints/'${data}'_v'${min_res}'a'${alpha}'_n'${base_noise_amp}'_'${aug_type}'/rand_n100_bin_r1.0x1.0x1.0'
    python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
    python eval_SSFID.py -s ${fake} -r ${ref} -g ${gpu_id}
    python eval_Div.py -s ${fake} -g ${gpu_id}
done

#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Acropolis_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Acropolis_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Acropolis_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Acropolis_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Cactus_r128s6.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cactus_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cactus_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cactus_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cactus_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cactus_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cactus_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cactus_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Canyon_r128s6.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Canyon_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Canyon_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Canyon_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Canyon_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Canyon_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Canyon_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Canyon_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Castle_r128s6.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Castle_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Castle_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Castle_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Castle_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Castle_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Castle_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Castle_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Cheese_r128s6.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cheese_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cheese_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cheese_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cheese_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cheese_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cheese_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Cheese_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/NaturalArch_r128s7.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/NaturalArch_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/NaturalArch_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/NaturalArch_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/NaturalArch_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Rock_r128s7.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Rock_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Rock_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Rock_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Rock_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Rock_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Rock_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Rock_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/SmallTown_r128s6.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/SmallTown_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/SmallTown_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/SmallTown_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/SmallTown_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/StoneStairs_r128s7.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/StoneStairs_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/StoneStairs_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/StoneStairs_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/StoneStairs_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Tree_r128s6.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Tree_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Tree_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Tree_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Tree_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Tree_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Tree_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Tree_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Vase_r128s7.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Vase_r128s7_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Vase_r128s7_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Vase_r128s7_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Vase_r128s7_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Vase_r128s7_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Vase_r128s7_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Vase_r128s7_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Wall_r128s6.h5'
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Wall_r128s6_v4a10_n0.1_cropping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Wall_r128s6_v4a10_n0.1_flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Wall_r128s6_v4a10_n0.1_rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Wall_r128s6_v4a10_n0.1_cropping,flipping/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Wall_r128s6_v4a10_n0.1_cropping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Wall_r128s6_v4a10_n0.1_flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
#
#fake='/scratch/hh1811/projects/CDDPM-v10/checkpoints/Wall_r128s6_v4a10_n0.1_cropping,flipping,rotation/rand_n100_bin_r1.0x1.0x1.0'
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}


#ref='/scratch/hh1811/projects/CDDPM-v10/data/Acropolis_r128s6.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Cactus_r128s6.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Canyon_r128s6.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Castle_r128s6.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Cheese_r128s6.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/NaturalArch_r128s7.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Rock_r128s7.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/SmallTown_r128s6.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/StoneStairs_r128s7.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Tree_r128s6.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Vase_r128s7.h5'
#ref='/scratch/hh1811/projects/CDDPM-v10/data/Wall_r128s6.h5'


#gpu_id=0
#
## LP-IoU and LP-F-score
#echo "python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}";
#python eval_LP.py -s ${fake} -r ${ref} -g ${gpu_id}
## SSFID
#echo "python eval_SSFID.py -s ${fake} -r ${ref} -g ${gpu_id}";
#python eval_SSFID.py -s ${fake} -r ${ref} -g ${gpu_id}
## Diversity (pairwise 1-IoU)
#echo "python eval_Div.py -s ${fake} -g ${gpu_id}";
#python eval_Div.py -s ${fake} -g ${gpu_id}
