import argparse
import copy
import numpy as np
import os
import sys
import torch
import yaml
from collections import OrderedDict
import matplotlib.pyplot as plt

from utils.data_utils import get_square
from datetime import datetime

# build arguments
parser = argparse.ArgumentParser()

# accessible arguments through bash scripts
########## General ##########
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--env', type=str, default='stow')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--stage', default='dy')
parser.add_argument('--skill_name', type=str, default='sweep') # push, insert, sweep
parser.add_argument('--tool_type', type=str, default='gripper_sym_rod_robot_v1_surf_nocorr_full')
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--dataset_seed', type=int, default=0)



########## Perception ##########
# ==================== TUNE at PERCEPTION ==================== #
parser.add_argument('--gen_rollout', type=int, default=1)
parser.add_argument('--surface_sample', type=int, default=0)
parser.add_argument('--farthest_point_sample', type=int, default=1)
parser.add_argument('--correspondance', type=int, default=0)
parser.add_argument('--rescontruct_from_pose', type=int, default=1)
parser.add_argument('--n_rollouts', type=int, default=100)

# ==================== TUNE at PERCEPTION ==================== #


########## Dynamics ##########
##### Train #####
# ==================== TUNE at TRAINING ==================== #
parser.add_argument('--batch_norm', type=int, default=0)
parser.add_argument('--train_set_ratio', type=float, default=1.0)
parser.add_argument('--valid', type=int, default=1)
parser.add_argument('--data_type', type=str, default='gt')
parser.add_argument('--loss_type', type=str, default='mse')# chamfer_emd, mse, l1
parser.add_argument('--rigid_motion', type=int, default=1)
parser.add_argument('--attn', type=int, default=0)
parser.add_argument('--full_repr', type=int, default=1)

parser.add_argument('--neighbor_radius', type=float, default=0.02)
parser.add_argument('--tool_neighbor_radius', type=float, default=0.05)
parser.add_argument('--motion_bound', type=float, default=-1)
parser.add_argument('--sequence_length', type=int, default=1)
parser.add_argument('--chamfer_weight', type=float, default=0.5)
parser.add_argument('--emd_weight', type=float, default=0.5)
parser.add_argument('--h_weight', type=float, default=0.0)
parser.add_argument('--loss_ord', type=int, default=2)
parser.add_argument('--dcd_alpha', type=int, default=50)

parser.add_argument('--check_tool_touching', type=int, default=0)
parser.add_argument('--tool_next_edge', type=int, default=0)
parser.add_argument('--dynamic_staic_edge', type=int, default=1) # for ablation study
parser.add_argument('--implicit', type=int, default=0)
parser.add_argument('--auxiliary_gripper_loss', type=int, default=0) # for ablation study

# ==================== TUNE at TRAINING ==================== #

parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--n_epoch', type=int, default=500)
parser.add_argument('--plateau_epoch_size', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--optimizer', default='Adam')

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--dy_model_path', type=str, default='')
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument('--ckp_per_iter', type=int, default=10000)

##### GNN #####
parser.add_argument('--n_his', type=int, default=1)
parser.add_argument('--p_rigid', type=float, default=1.0)
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--time_gap', type=int, default=1)

##### Eval #####
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--eval_epoch', type=int, default=-1)
parser.add_argument('--eval_iter', type=int, default=-1)
# parser.add_argument('--n_rollout', type=int, default=0)



# precoded arguments
def gen_args():
    args = parser.parse_args()

    args.data_names = ['positions', 'obj_id', 'attr', 'wall_pos_y', 'obs', 'tool_repr']

    args.physics_param_range = (-5., -5.)
    args.scene_params = np.array([1, 1, 0]) # n_instance, gravity, draw_mesh


    if 'full' in args.tool_type:
        args.full_repr = 1

    args = gen_args_env(args)
    args = gen_args_pipeline(args)

    return args


def gen_args_env(args):
    ##### path ######
    # training data
    args.dy_data_path = f'data/{args.data_type}/stow_{args.skill_name}'
    # traing output
    args.dy_out_path =  f'dump_sample_size/dynamics/dump_{args.skill_name}'
    
    # traing output
    args.demo_path =  f'dump/demos'

    args.target_path = f'target_shapes/sim'
    
    # ROS package
    args.ros_pkg_path = "/home/haonan/Projects/catkin_ws/src/deformable_ros"
    # args.ros_pkg_path = "/home/haochen/catkin_ws/src/deformable_ros"
    # tool models
    args.tool_geom_path = "geometries/tools"
    # tool repr models
    args.tool_repr_path = "geometries/reprs"

    args.data_time_step = 1

    ##### camera ######
    args.depth_optical_frame_pose = [0, 0, 0, 0.5, -0.5, 0.5, -0.5]

    ##### robot #####
    args.ee_fingertip_T_mat = np.array([[1., 0., 0, 0], [0.,  1.,  0., 0], [0., 0.,  1, 0.0], [0, 0, 0, 1]])

    # 5cm: 0.042192, 7cm: 0.052192, 8cm: 0.057192
    args.tool_center_z = 0.06472

    mid_point_sim = np.array([0.5, 0.1, 0.5])
    # robocraft
    mid_point_robot = np.array([0.43, -0.01, 0.1])
    
    args.mid_point = mid_point_robot
    args.axes = [0, 1, 2]

    args.show_box = 0

    ##### attribute #####
    args.attr_dim = 2 # dynamic or static, valid or dummy


    ##### tools #####
    # args.tool_neighbor_radius = [float(x) for x in args.tool_neighbor_radius.split('+')]
    args.sim_kwargs = {'robots': 'Kinova3',
                        'controller_configs': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1,
                                                'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                                                'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                                                'kp': 150, 'damping_ratio': 1,
                                                'impedance_mode': 'fixed',
                                                'kp_limits': [0, 300],
                                                'damping_ratio_limits': [0, 10],
                                                'position_limits': None,
                                                'orientation_limits': None,
                                                'uncouple_pos_ori': True,
                                                'control_delta': False,
                                                'interpolation': None,
                                                'ramp_ratio': 0.2},
                        'reward_shaping': False,
                        'camera_names': [ 'agentview', 'cam_1', 'paper', 'cam_slide', 'cam_push'],# 'cam_1',  'cam_2', 'cam_3', 'cam_4',
                        'camera_heights': 800,
                        'camera_widths': 1280}
    
    args.max_skill_calls = 1
    
    args.max_shelf_objs = 4
    args.max_n_instance = 4 + args.max_shelf_objs

    args.add_corners_points = 1
    args.num_corners_points = 8 if args.add_corners_points else 0

    args.n_box_particle = 56 + args.num_corners_points
    args.dummy_box_state = np.zeros((args.n_box_particle,3))


    args.crop_bound = [[-0.25, -0.8, 0.8], [0.5, 0.32, 1.5]]
    
    args.skill_config = dict(
        skills=['push', 'insert', 'sweep'],
        push_config={'param_bounds': [[-0.1, 0.0, 0.22], [-0.03, 0.05, 0.26]],  # x, y, distance
                    'hover_z': 0.95,
                    'push_z': 0.818,
                    'push_quat': [-0.707, 0.707, 0., 0.], },  # scalar last
        insert_config={'param_bounds': [[0.0, 0.05, 0.8, 0.6], [0.01, 0.1, 0.95, 0.9]],  # offset in y direction, relative height, sliding percentage, quat percentage
                    'quat_bounds': [[-0.5, -0.5, -0.5, -0.5], [-0.707, 0, -0.707, 0.]],
                    'reach_offset': [-0.1, -0.088, 0.0], # -0.13, -0.067, 0.0
                    'grasp_offset': [0.02 , -0.088, 0.0], # 0.01, -0.067, 0.0
                    'grasp_quat': [-0.5, 0.5, -0.5, 0.5],  # scalar last
                    'lift_z': 1.1,
                    'stand_quat': [-0.707, 0, -0.707, 0.],  # scalar last
                    'insert_start_x': 0.06,
                    'insert_target_x': 0.29,
                    'shelf_y_range': [-0.15, 0.105],
                    },  
        sweep_config={'param_bounds': [[0.0, 0.1, 1], [0.01, 0.15, 1.4]],  # offset in y direction, sweep height, sliding percentage
                    'sweep_quat': [-0.5, 0.5, -0.5, 0.5],  # scalar last
                    'push_x': 0.08,
                    'sweep_x': 0.35,
                    'sweep_z': 0.89,
                    'shelf_y_range': [-0.06, 0.063],
                    },  

    )
    
    if args.skill_name=='push':
        args.plot_radius = 0.45
    else:
        args.plot_radius = 0.3
    


    args.evenly_spaced = True

    args.show_src_img = True
    
    args.env_config = None
    
    
    ##### object ######

    args.n_particles_per_object = generate_n_particles_per_object(args, args.max_shelf_objs)
    args.n_particles_per_object_clean = OrderedDict()
    for k, v in args.n_particles_per_object.items():
        if 'shelf' in k:
            args.n_particles_per_object_clean['shelf'] = args.n_particles_per_object_clean.get('shelf', 0) + v
        else:
            args.n_particles_per_object_clean[k] = v

     # movable, fixed, tool
    args.n_particles_type = [sum([v for k, v in args.n_particles_per_object.items() if 'cube' in k]), 
                            sum([v for k, v in args.n_particles_per_object.items() if 'table' in k or 'shelf' in k]),
                            sum([v for k, v in args.n_particles_per_object.items() if 'gripper' in k])]
    args.n_particles = args.n_particles_type[0] 
    # args.dummy_particle_dim = args.n_particles_type[1] 
    args.floor_dim = args.n_particles_type[1] # exclude tool
    
    args.gripper_dim = args.n_particles_per_object['gripper']
    
    args.n_instance = args.max_shelf_objs + 4 # shelf are one instance

    args.tool_dim =  [int(list(args.n_particles_per_object.values())[-1]/2), 
                                    int(list(args.n_particles_per_object.values())[-1]/2)] 
    args.gripper_size = np.array([0.017, 0.007, 0.038])
    return args


def gen_args_pipeline(args):
    ##### dynamics #####
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")        

    args.use_distance = 0
    if args.use_distance:
        args.state_dim = 6
    else: 
        args.state_dim = 3
    args.mean_p = np.array([ 0.2099, -0.0562,  0.9462])
    args.std_p = np.array([ 0.2096, 0.2083, 0.1277])

    args.mean_d = np.array([-0.0188, -0.0350, -0.0711])
    args.std_d = np.array([0.0566, 0.0572, 0.0795])

    args.nf_relation = 150
    args.nf_particle = 150
    args.nf_pos = 150
    args.nf_memory = 150
    args.mem_nlayer = 2
    args.nf_effect = 150
    

    return args


def update_dy_args(args, dy_args_dict):
    args.surface_sample = dy_args_dict['surface_sample']
    args.rigid_motion = dy_args_dict['rigid_motion']
    args.attn = dy_args_dict['attn']
    args.neighbor_radius = dy_args_dict['neighbor_radius']
    args.full_repr = dy_args_dict['full_repr']
    args.tool_dim = dy_args_dict['tool_dim']
    args.tool_type = dy_args_dict['tool_type']
    args.tool_neighbor_radius = dy_args_dict['tool_neighbor_radius']
    args.data_type = dy_args_dict['data_type']
    args.n_his = dy_args_dict['n_his']
    args.time_gap = getattr(dy_args_dict, 'time_gap', 1) # for backward compatibility
    args.state_dim = dy_args_dict['state_dim']
    args.mean_p = dy_args_dict['mean_p']
    args.std_p = dy_args_dict['std_p']

    return args


def print_args(args):
    for key, value in dict(sorted(args.__dict__.items())).items():
        if not key in ['floor_state', 'tool_center', 'tool_geom_mapping', 'tool_neighbor_radius_dict', 'tool_full_repr_dict']:
            print(f'{key}: {value}')

def generate_n_particles_per_object(args, num_obj_on_shelf):
    n_particles_per_object = OrderedDict()

    # Add fixed objects
    n_particles_per_object["cubeA"] = args.n_box_particle 
    # Add cubeB objects based on num_obj_on_shelf
    for i in range(num_obj_on_shelf):
        n_particles_per_object[f"cubeB_{i}"] = args.n_box_particle 

    n_particles_per_object["table"] = 63 + args.num_corners_points
    n_particles_per_object["shelf_bottom"] = 18 + args.num_corners_points
    n_particles_per_object["shelf_right"] = 15 + args.num_corners_points
    n_particles_per_object["shelf_left"] = 15 + args.num_corners_points
    n_particles_per_object["gripper"] = 16 + args.num_corners_points*2


    return n_particles_per_object

