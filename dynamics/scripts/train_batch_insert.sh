#!/bin/bash

# Define your hyperparameters
declare -a dynamic_staic_edges=("0" "1")
declare -a auxiliary_gripper_losses=("0" "1")

# Define your default parameters
skill_name="insert"
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
neighbor_radius="0.15"
tool_neighbor_radius="0.175"
pstep="2"
check_tool_touching="0"
tool_next_edge="0"
random_seed="0"
eval="0"

for dynamic_staic_edge in "${dynamic_staic_edges[@]}"
do
    for auxiliary_gripper_loss in "${auxiliary_gripper_losses[@]}"
    do
        sbatch dynamics/scripts/train.sh $skill_name $data_type $debug $loss_type $n_his $sequence_length $time_gap $rigid_motion $attn $chamfer_weight $emd_weight $neighbor_radius $tool_neighbor_radius $pstep $check_tool_touching $tool_next_edge $dynamic_staic_edge $auxiliary_gripper_loss $random_seed $eval

    done
done
