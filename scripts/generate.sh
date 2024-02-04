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
#conda activate torch2.0_cuda11.8
conda activate sinwavelet

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
    python main.py test --tag ${tag} -g 0 --n_samples 100 --mode rand --resize 1.0 1.0 1.0 --upsample 1
done


#tag=${data}'_v'${min_voxels}'d'${max_depth}'a'${alpha}'_'${alpha_policy}'_allnoise'

#tag='Acropolis_r128s6_v4a10_n0.1_cropping'
#tag='Acropolis_r128s6_v4a10_n0.1_flipping'
#tag='Acropolis_r128s6_v4a10_n0.1_rotation'
#tag='Acropolis_r128s6_v4a10_n0.1_cropping,flipping'
#tag='Acropolis_r128s6_v4a10_n0.1_cropping,rotation'
#tag='Acropolis_r128s6_v4a10_n0.1_flipping,rotation'
#tag='Acropolis_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#tag='Cactus_r128s6_v4a10_n0.1_cropping'
#tag='Cactus_r128s6_v4a10_n0.1_flipping'
#tag='Cactus_r128s6_v4a10_n0.1_rotation'
#tag='Cactus_r128s6_v4a10_n0.1_cropping,flipping'
#tag='Cactus_r128s6_v4a10_n0.1_cropping,rotation'
#tag='Cactus_r128s6_v4a10_n0.1_flipping,rotation'
#tag='Cactus_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#tag='Canyon_r128s6_v4a10_n0.1_cropping'
#tag='Canyon_r128s6_v4a10_n0.1_flipping'
#tag='Canyon_r128s6_v4a10_n0.1_rotation'
#tag='Canyon_r128s6_v4a10_n0.1_cropping,flipping'
#tag='Canyon_r128s6_v4a10_n0.1_cropping,rotation'
#tag='Canyon_r128s6_v4a10_n0.1_flipping,rotation'
#tag='Canyon_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#tag='Castle_r128s6_v4a10_n0.1_cropping'
#tag='Castle_r128s6_v4a10_n0.1_flipping'
#tag='Castle_r128s6_v4a10_n0.1_rotation'
#tag='Castle_r128s6_v4a10_n0.1_cropping,flipping'
#tag='Castle_r128s6_v4a10_n0.1_cropping,rotation'
#tag='Castle_r128s6_v4a10_n0.1_flipping,rotation'
#tag='Castle_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#tag='Cheese_r128s6_v4a10_n0.1_cropping'
#tag='Cheese_r128s6_v4a10_n0.1_flipping'
#tag='Cheese_r128s6_v4a10_n0.1_rotation'
#tag='Cheese_r128s6_v4a10_n0.1_cropping,flipping'
#tag='Cheese_r128s6_v4a10_n0.1_cropping,rotation'
#tag='Cheese_r128s6_v4a10_n0.1_flipping,rotation'
#tag='Cheese_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#tag='NaturalArch_r128s7_v4a10_n0.1_cropping'
#tag='NaturalArch_r128s7_v4a10_n0.1_flipping'
#tag='NaturalArch_r128s7_v4a10_n0.1_rotation'
#tag='NaturalArch_r128s7_v4a10_n0.1_cropping,flipping'
#tag='NaturalArch_r128s7_v4a10_n0.1_cropping,rotation'
#tag='NaturalArch_r128s7_v4a10_n0.1_flipping,rotation'
#tag='NaturalArch_r128s7_v4a10_n0.1_cropping,flipping,rotation'

#tag='Rock_r128s7_v4a10_n0.1_cropping'
#tag='Rock_r128s7_v4a10_n0.1_flipping'
#tag='Rock_r128s7_v4a10_n0.1_rotation'
#tag='Rock_r128s7_v4a10_n0.1_cropping,flipping'
#tag='Rock_r128s7_v4a10_n0.1_cropping,rotation'
#tag='Rock_r128s7_v4a10_n0.1_flipping,rotation'
#tag='Rock_r128s7_v4a10_n0.1_cropping,flipping,rotation'

#tag='SmallTown_r128s6_v4a10_n0.1_cropping'
#tag='SmallTown_r128s6_v4a10_n0.1_flipping'
#tag='SmallTown_r128s6_v4a10_n0.1_rotation'
#tag='SmallTown_r128s6_v4a10_n0.1_cropping,flipping'
#tag='SmallTown_r128s6_v4a10_n0.1_cropping,rotation'
#tag='SmallTown_r128s6_v4a10_n0.1_flipping,rotation'
#tag='SmallTown_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#tag='StoneStairs_r128s7_v4a10_n0.1_cropping'
#tag='StoneStairs_r128s7_v4a10_n0.1_flipping'
#tag='StoneStairs_r128s7_v4a10_n0.1_rotation'
#tag='StoneStairs_r128s7_v4a10_n0.1_cropping,flipping'
#tag='StoneStairs_r128s7_v4a10_n0.1_cropping,rotation'
#tag='StoneStairs_r128s7_v4a10_n0.1_flipping,rotation'
#tag='StoneStairs_r128s7_v4a10_n0.1_cropping,flipping,rotation'

#tag='Tree_r128s6_v4a10_n0.1_cropping'
#tag='Tree_r128s6_v4a10_n0.1_flipping'
#tag='Tree_r128s6_v4a10_n0.1_rotation'
#tag='Tree_r128s6_v4a10_n0.1_cropping,flipping'
#tag='Tree_r128s6_v4a10_n0.1_cropping,rotation'
#tag='Tree_r128s6_v4a10_n0.1_flipping,rotation'
#tag='Tree_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#tag='Vase_r128s7_v4a10_n0.1_cropping'
#tag='Vase_r128s7_v4a10_n0.1_flipping'
#tag='Vase_r128s7_v4a10_n0.1_rotation'
#tag='Vase_r128s7_v4a10_n0.1_cropping,flipping'
#tag='Vase_r128s7_v4a10_n0.1_cropping,rotation'
#tag='Vase_r128s7_v4a10_n0.1_flipping,rotation'
#tag='Vase_r128s7_v4a10_n0.1_cropping,flipping,rotation'

#tag='Wall_r128s6_v4a10_n0.1_cropping'
#tag='Wall_r128s6_v4a10_n0.1_flipping'
#tag='Wall_r128s6_v4a10_n0.1_rotation'
#tag='Wall_r128s6_v4a10_n0.1_cropping,flipping'
#tag='Wall_r128s6_v4a10_n0.1_cropping,rotation'
#tag='Wall_r128s6_v4a10_n0.1_flipping,rotation'
#tag='Wall_r128s6_v4a10_n0.1_cropping,flipping,rotation'

#python main.py test --tag ${tag} -g 0 --n_samples 100 --mode rand --resize 2.0 1.0 1.0 --upsample 1

#python main.py test --tag ${tag} -g 0 --n_samples 100 --mode rand --resize 1.0 1.0 1.0 --upsample 1