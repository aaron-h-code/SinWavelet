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
#SBATCH -q cair

#SBATCH --mail-type=END
#SBATCH --mail-user=hh1811@nyu.edu

source ~/.bashrc
#conda activate torch2.0_cuda11.8
conda activate sinwavelet

min_res=4
alpha=10
base_noise_amp=0.1

#aug_type='cropping'
#aug_type='flipping'
#aug_type='rotation'
#aug_type='cropping,flipping'
#aug_type='cropping,rotation'
#aug_type='flipping,rotation'
#aug_type='cropping,flipping,rotation'

aug_types=('cropping' 'flipping' 'rotation' 'cropping,flipping,rotation')

batch_size=2

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

data_path='data/'${data}'.h5'

for aug_type in "${aug_types[@]}"; do
    tag=${data}'_v'${min_res}'a'${alpha}'_n'${base_noise_amp}'_'${aug_type}
    # Arguments
    # --tag: experiment-tag
    # --data_path: path-to-processed-h5-data
    python main.py train --tag ${tag} -s ${data_path} --min_res ${min_res} --alpha ${alpha} --base_noise_amp ${base_noise_amp} --batch_size ${batch_size} --aug_type ${aug_type} -g 0
done


#tag='dev'

#python main.py train --tag acropolis_r256s8 -s data/acropolis_r256s8.h5 --max_depth 5 --timesteps 5 -g 0