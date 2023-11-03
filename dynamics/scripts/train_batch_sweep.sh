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
#SBATCH --gres=gpu:a100:2
#SBATCH --export=ALL
 
source /projects/haonan2/miniconda3/bin/activate
conda init bash
scl enable devtoolset-9 bash
conda activate py38

echo STARTING `date`

# Define your hyperparameters
declare -a dynamic_staic_edges=("0" "1")
declare -a auxiliary_gripper_losses=("0" "1")

# Define your default parameters
skill_name="sweep"
stage="dy"
data_type="gt"
loss_type="mse"
debug="0"
n_his="1"
sequence_length="1"
time_gap="1"
rigid_motion="1"
attn="0"
chamfer_weight="0.5"
emd_weight="0.5"
neighbor_radius="0.175"
tool_neighbor_radius="0.175"
pstep="2"
check_tool_touching="0"
tool_next_edge="0"
random_seed="2"
eval="0"

cd /projects/haonan2/Stowing

for dynamic_staic_edge in "${dynamic_staic_edges[@]}"
do
    for auxiliary_gripper_loss in "${auxiliary_gripper_losses[@]}"
    do
        python dynamics/train.py \
            --stage $stage \
            --skill_name $skill_name \
            --data_type $data_type \
            --debug $debug \
            --loss_type $loss_type \
            --n_his $n_his \
            --sequence_length $sequence_length \
            --time_gap $time_gap \
            --rigid_motion $rigid_motion \
            --attn $attn \
            --chamfer_weight $chamfer_weight \
            --emd_weight $emd_weight \
            --neighbor_radius $neighbor_radius \
            --tool_neighbor_radius $tool_neighbor_radius \
            --pstep $pstep \
            --check_tool_touching $check_tool_touching \
            --tool_next_edge $tool_next_edge \
            --dynamic_staic_edge $dynamic_staic_edge \
            --auxiliary_gripper_loss $auxiliary_gripper_loss \
            --random_seed $random_seed \
            --eval $eval
    done
done
