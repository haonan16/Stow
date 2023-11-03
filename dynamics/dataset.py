
import glob
import numpy as np
import os
from collections import OrderedDict
import torch.nn.functional as F

import torch

from torch.utils.data import Dataset
from utils.data_utils import *
# from perception.sample import pcd_dis_wall

# def get_single_obj_state(obj_name, obj_state, id, args, num_objects):
#     '''helper function to get the state of a single object'''
#     obj_gt_state = torch.tensor(obj_state, device=args.device, dtype=torch.float32)
#     obj_id = torch.zeros((obj_gt_state.shape[0], num_objects), device=args.device).float()
#     obj_id[:, id] = 1
#     attr = torch.zeros((obj_gt_state.shape[0], args.attr_dim), device=args.device).float()
#     # attribute 0: [movable, fixed]
#     # attribute 1: [floor]
#     # attribute 2: [gripper]
#     if 'cube' in obj_name or 'gripper' in obj_name: 
#         attr[..., 0] = 1 # movable
#     else:
#         attr[..., 0] = 0 # fixed
#     return obj_gt_state, obj_id, attr


class GNNDataset(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(args.dy_data_path, phase)
        self.stat_path = os.path.join('data/stats.h5')
        self.dataset_len = 0

        
        vid_path_list = sorted(glob.glob(os.path.join(self.data_dir, '*')))
        if phase == 'train':
            vid_path_list = vid_path_list[:int(args.train_set_ratio * len(vid_path_list))]
            print(f"Using {len(vid_path_list)} videos for training")

        self.state_data_list = []
        self.tool_data_list = []
        self.action_data_list = []
        self.obj_data_list = []
        self.attr_data_list = []
        n_frames_min = float('inf')
        self.load_stats()
        for gt_vid_path in vid_path_list:
            frame_start = 0
            n_frames = len(glob.glob(os.path.join(gt_vid_path, '*.h5')))
            n_frames_min = min(n_frames, n_frames_min)
            gt_state_list = []
            tool_repr_list = []
            gt_action_list = []
            gt_obj_list = []
            gt_attr_list = []
            
            for i in range(n_frames):

                # gt_frame_data = np.load(f"{gt_vid_path}/{i:03d}_pcd.npy", allow_pickle=True).item()
                # obj_gt_state, obj_id, attr = zip(*[get_single_obj_state(obj_name, obj_state, id, args, self.num_objects) for id, (obj_name, obj_state) in enumerate(gt_frame_data.items())])
                
                # gt_state = torch.vstack(obj_gt_state)
                frame_data = load_data(args.data_names, f"{gt_vid_path}/{i:03d}.h5")
                obj_gt_state, obj_id, attr, pcd_dis_wall, _, tool_repr = frame_data[:6]

                # if args.auxiliary_gripper_loss:
                #     pass
                # else:
                #     obj_gt_state[-self.args.gripper_dim:] = tool_repr
                gt_state = torch.from_numpy(obj_gt_state).to(args.device).float()
                tool_repr = torch.from_numpy(tool_repr).to(args.device).float()

                if args.use_distance:
                    # pcd_dis_wall = load_data(args.data_names, f"{gt_vid_path}/{i:03d}.h5")[0]
                    pcd_dis_wall = np.clip(pcd_dis_wall, -args.neighbor_radius, args.neighbor_radius)
                    pcd_dis_wall_dynamic = np.zeros_like(pcd_dis_wall)
                    pcd_dis_wall_dynamic[:self.args.n_particles] = pcd_dis_wall[:self.args.n_particles]
                    gt_state = torch.cat((gt_state, torch.from_numpy(pcd_dis_wall).to(args.device)), dim=1).float()
                    
                gt_state_list.append(gt_state)
                tool_repr_list.append(tool_repr)
                # gt_action_list.append(torch.from_numpy(tool_repr).to(args.device).float())
                gt_obj_list.append(torch.from_numpy(obj_id).to(args.device).float())
                
                # attr = torch.vstack(attr)
                gt_attr_list.append(torch.from_numpy(attr).to(args.device).float())
                
            
            self.obj_data_list.append(torch.stack(gt_obj_list))
            self.attr_data_list.append(torch.stack(gt_attr_list))
            # self.action_data_list.append(torch.stack(gt_action_list))
            
            state_seq_list = []
            tool_seq_list = []
            for i in range(frame_start, n_frames - args.time_gap * (args.n_his + args.sequence_length - 1)):
                state_seq = []
                tool_seq = []
                # history frames
                for j in range(i, i + args.time_gap * (args.n_his - 1) + 1, args.time_gap):
                    # print(f'history: {j}')
                    state_seq.append(gt_state_list[j])
                    tool_seq.append(tool_repr_list[j])

                # frames to predict
                for j in range(i + args.time_gap * args.n_his, 
                    i + args.time_gap * (args.n_his + args.sequence_length - 1) + 1, args.time_gap):
                    # print(f'predict: {j}')
                    state_seq.append(gt_state_list[j])
                    tool_seq.append(tool_repr_list[j])

                self.dataset_len += 1
                state_seq_list.append(torch.stack(state_seq))
                tool_seq_list.append(torch.stack(tool_seq))

            self.state_data_list.append(state_seq_list)
            self.tool_data_list.append(tool_seq_list)

        print(f"{phase} -> number of sequences: {self.dataset_len}")
        print(f"{phase} -> minimum number of frames: {n_frames_min}")


    def __len__(self):
        # Each data point consists of a sequence
        return self.dataset_len


    def load_stats(self):
        pass
        # print("Loading stat from %s ..." % self.stat_path)
        # self.stat = load_data(self.args.data_names[:1], self.stat_path)
        # self.num_objects = 6 # it includes the gripper

    def cal_stats(self):
        '''calculate the mean and std of the objects' states and tool motion'''
        self.n_p = self.args.n_particles_type[0]

        mean_p = torch.mean(torch.stack([skill_idx[:,:self.n_p] for rollout in self.state_data_list for skill_idx in rollout]), dim=(0,1,2))
        std_p = torch.std(torch.stack([skill_idx[:,:self.n_p] for rollout in self.state_data_list for skill_idx in rollout]), dim=(0,1,2))

        tool_start_idx = sum(self.args.n_particles_type) - self.args.n_particles_type[-1]
        mean_d = torch.mean(torch.stack([skill_idx[-1,tool_start_idx:] - skill_idx[0,tool_start_idx:] for rollout in self.state_data_list for skill_idx in rollout]), dim=(0,1))
        std_d = torch.std(torch.stack([skill_idx[-1,tool_start_idx:] - skill_idx[0,tool_start_idx:] for rollout in self.state_data_list for skill_idx in rollout]), dim=(0,1))

    # @profile
    def __getitem__(self, idx):
        args = self.args

        idx_curr = idx
        idx_vid = 0
        offset = len(self.state_data_list[0])
        while idx_curr >= offset:
            idx_curr -= offset
            idx_vid = (idx_vid + 1) % len(self.state_data_list)
            offset = len(self.state_data_list[idx_vid])

        state_seq = self.state_data_list[idx_vid][idx_curr][:args.n_his+args.sequence_length]
        tool_seq = self.tool_data_list[idx_vid][idx_curr][:args.n_his+args.sequence_length]
        # action_seq = self.action_data_list[idx_vid][idx_curr: idx_curr  * args.time_gap * \
        #     (args.n_his + args.sequence_length - 1) + 1: args.data_time_step]
        obj_seq = self.obj_data_list[idx_vid][idx_curr]
        attr_seq = self.attr_data_list[idx_vid][idx_curr]
        return state_seq, obj_seq, attr_seq, tool_seq
