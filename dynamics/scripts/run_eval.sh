#!/usr/bin/env bash

skill_name="insert"
dy_model_path="dy_gt_nr=0.05_tnr=0.05_0.05_his=1_seq=1_time_gap=1_chamfer_emd_0.5_0.5_rm=1_valid_Apr-12-08:54:36/net_best.pth"

bash dynamics/scripts/eval.sh $skill_name $dy_model_path
