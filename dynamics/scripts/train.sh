#!/bin/bash

#SBATCH --job-name="stow"
#SBATCH --output="/projects/haonan2/stow_out/stow.%j.%N.out"
#SBATCH --error="/projects/haonan2/stow_out/stow.%j.%N.err"
#SBATCH --partition=x86
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=16
#SBATCH --threads-per-core=2
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:a100:1
#SBATCH --export=ALL
 
source /projects/haonan2/miniconda3/bin/activate

#cd ~
conda init bash

scl enable devtoolset-9 bash
	
conda activate py38

echo STARTING `date`

#srun hostname

cd /projects/haonan2/Stowing

python dynamics/train.py \
	--stage dy \
	--skill_name $1 \
	--data_type $2 \
	--debug $3 \
	--loss_type $4 \
	--n_his $5 \
	--sequence_length $6 \
	--time_gap $7 \
	--rigid_motion $8 \
	--attn $9 \
	--chamfer_weight ${10} \
	--emd_weight ${11} \
	--neighbor_radius ${12} \
	--tool_neighbor_radius ${13} \
	--pstep ${14} \
	--check_tool_touching ${15} \
	--tool_next_edge ${16} \
	--dynamic_staic_edge ${17} \
	--auxiliary_gripper_loss ${18} \
	--random_seed ${19} \
	--eval 1 