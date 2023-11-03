skill_name="insert"
data_type="gt"
debug=0
loss_type="chamfer_emd" # "chamfer_emd"
n_his=1
sequence_length=1
time_gap=1
rigid_motion=1
attn=0
chamfer_weight=0.5
emd_weight=0.5
neighbor_radius=0.15
tool_neighbor_radius=0.15
pstep=2
check_tool_touching=0
tool_next_edge=0
dynamic_staic_edge=0

bash dynamics/scripts/train.sh $skill_name $data_type $debug $loss_type $n_his $sequence_length $time_gap \
                                $rigid_motion $attn $chamfer_weight $emd_weight $neighbor_radius $tool_neighbor_radius \
                                $pstep $check_tool_touching $tool_next_edge