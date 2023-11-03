import numpy as np
import os
import sys
import torch

torch.set_default_tensor_type(torch.FloatTensor)

from dynamics.dy_utils import *
from dynamics.model import Model
from perception.pcd_utils import upsample
from pytorch3d.transforms import *
from transforms3d.axangles import axangle2mat
# from sim import Simulator
from config.config import gen_args
from utils.data_utils import *
from utils.loss import *
from utils.visualize import *


class GNN(object):
    def __init__(self, args, model_path):
        # load dynamics model
        self.args = args
        set_seed(args.random_seed)

        self.model = Model(args)
        # print("model_kp #params: %d" % count_parameters(self.model))

        self.device = args.device
        pretrained_dict = torch.load(model_path, map_location=self.device)

        model_dict = self.model.state_dict()
        
        # only load parameters in dynamics_predictor
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() \
            if 'dynamics_predictor' in k and k in model_dict
        }
        
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.model.eval()

        self.model = self.model.to(self.device)


    def prepare_shape(
        self,
        batch_size,
        state_cur,      # [N, state_dim]
        # init_tool_pose_seqs, # [B, n_grip, n_shape, 14]
        # act_seqs,       # [B, n_grip, n_steps, 12]
        attr_cur, 
        rot_seqs,       # [B, n_grip]
    ):
        # Convert non-tensor inputs to PyTorch tensors
        # init_tool_pose_seqs = torch.as_tensor(init_tool_pose_seqs).float().to(self.device)
        # act_seqs = torch.as_tensor(act_seqs).float().to(self.device)
        state_cur = torch.as_tensor(state_cur).float().to(self.device)
        attr_cur = torch.as_tensor(attr_cur).float().to(self.device)
        rot_seqs = torch.as_tensor(rot_seqs).float().to(self.device) if rot_seqs is not None else None

        # init_tool_pose_seqs = init_tool_pose_seqs.float().to(self.device)
        # act_seqs = act_seqs.float().to(self.device)
        
        # Reshape tensors if necessary
        B = batch_size
        state_cur = expand(B, state_cur if len(state_cur.shape) == 3 else state_cur.unsqueeze(0)).to(self.device)
        attr_cur = expand(B, attr_cur if len(attr_cur.shape) == 3 else attr_cur.unsqueeze(0)).to(self.device)

        # if self.args.state_dim == 6:
        #     floor_state = torch.tensor(np.concatenate((self.args.floor_state, self.args.floor_normals), axis=1))
        # else:
        #     floor_state = torch.tensor(self.args.floor_state)
        # floor_state = expand(B, floor_state.float().unsqueeze(0)).to(self.device)

        return rot_seqs, state_cur, None, attr_cur


    def rotate_state(self, state_cur, theta, center_xy=[0.429, -0.008]):
        B = state_cur.shape[0]
        zero_padding = torch.zeros_like(theta, device=self.device)
        center_xy = torch.tensor(center_xy, device=self.device, dtype=torch.float32)
        center = torch.cat([torch.tile(center_xy, (B, 1, 1)), torch.mean(state_cur, dim=1)[:, 2:3].unsqueeze(1)], dim=2)

        state_rot = euler_angles_to_matrix(torch.stack((zero_padding, zero_padding, theta), dim=-1), 'XYZ')
        state_cur_new = (state_cur - center).bmm(state_rot) + center

        return state_cur_new


    # @profile
    def rollout(
        self,
        state_cur,      # [N, state_dim]
        action_his,     # [B, n_his, tool_dim, 3]
        # init_tool_pose_seqs, # [B, 1, n_particle_tool, 3]
        # act_seqs,       # [B, 1, n_steps, 12]
        attr_cur,      # [N, attr_dim]
        rot_seqs=None,       # [B, 1]
    ):
        visualize = False
        # reshape the tensors
        B = action_his.shape[0]
        action_his = torch.as_tensor(action_his).float().to(self.device)
        rot_seqs, state_cur, floor_state, attr_cur= self.prepare_shape(B, state_cur, attr_cur, rot_seqs)

        N = self.args.n_particles + sum(self.args.tool_dim) + self.args.floor_dim
        memory_init = self.model.init_memory(B, N)
        group_gt = get_env_group(self.args, B)

        if self.args.batch_norm:
            mean_p, std_p, _, _ = compute_stats(self.args, state_cur)
        else:
            mean_p = torch.FloatTensor(self.args.mean_p).to(self.device)
            std_p = torch.FloatTensor(self.args.std_p).to(self.device)

        mean_d = torch.FloatTensor(self.args.mean_d).to(self.device)
        std_d = torch.FloatTensor(self.args.std_d).to(self.device)
        
        stats = [mean_p, std_p, mean_d, std_d]
        # print(stats)

        tool_start_idx = self.args.n_particles + self.args.floor_dim

        # rollout
        states_pred_list = []
        attn_mask_list = []
        p2t_rels_list = [[] for _ in range(B)]
        pred_transformations = []  # Added line to initialize transformation list

        for i in range(action_his.shape[1] -1): 
            rels_list_prev = None
            # tool_pos_list = []
            action_his_i = action_his[:, i:i+2, :, :]
                    
            if i == 0:
                state_cur_dynamic = state_cur[:, :self.args.n_particles+self.args.floor_dim, :]
            else:
                state_cur_dynamic = pred_pos_p 

            # if rot_seqs is not None:
            #     state_cur_dynamic[...,:3] = self.rotate_state(state_cur_dynamic[...,:3], rot_seqs[:, i])
            if not self.args.use_distance:
                state_cur_dynamic = torch.cat([state_cur_dynamic[...,:3], action_his_i[:,0]], dim=1)
            else:
                pcd_dis_wall_dynamic = torch.zeros_like(state_cur[...,-3:])
                pcd_dis_wall = torch.clip(state_cur_dynamic[...,-3:], -self.args.neighbor_radius, self.args.neighbor_radius)
                pcd_dis_wall_dynamic[:, :self.args.n_particles] = pcd_dis_wall[:, :self.args.n_particles]
                state_cur_dynamic = torch.cat((torch.cat([state_cur_dynamic[...,:3], action_his_i[:,0]], dim=1), pcd_dis_wall_dynamic),dim=2)
            
            # action_his = torch.cat([torch.cat(x, dim=1) for x in action_his_list], dim=2)

            # import pdb; pdb.set_trace()
            Rr_curs, Rs_curs, rels_list_batch, p2t_rels_list_batch = prepare_input(
                self.args, state_cur_dynamic[:, :, :3].detach().cpu().numpy(), action_his[:,i+1].detach().cpu().numpy(),  rels_list_prev=rels_list_prev, device=self.device)
            rels_list_prev = rels_list_batch
            # print(attrs.shape, state_cur.shape, Rr_curs.shape, Rs_curs.shape)
            inputs = [attr_cur, state_cur_dynamic, action_his_i, Rr_curs, Rs_curs, memory_init, group_gt, stats]
            pred_pos_p, _, nrm_attn, pred_tool_p, pred_transformation = self.model.predict_dynamics(inputs)
            
            pred_pos_p = torch.cat((pred_pos_p, state_cur[:, self.args.n_particles:self.args.n_particles+self.args.floor_dim, :3]), dim=1)
            
            states_pred_list.append(pred_pos_p)
            
            if visualize:
                action_vis = torch.cat([action_his_i[:,k,:] for k in range(action_his_i.shape[1])], dim=1)
                visualize_o3d(self.args, [state_cur.cpu().numpy().squeeze(), action_vis.cpu().numpy().squeeze()], title='action pcd')

            attn_mask_list.append(nrm_attn)
            for k in range(B):
                p2t_rels_list[k].append(p2t_rels_list_batch[k])

            pred_transformations.append(pred_transformation)  # Added line to store each transformation

        states_pred_array = torch.stack(states_pred_list, dim=1)
        attn_mask_arry = torch.stack(attn_mask_list, dim=1)
        # print(f"torch mem allocated: {torch.cuda.memory_allocated()}")

        return states_pred_array, attn_mask_arry, p2t_rels_list, pred_transformations

