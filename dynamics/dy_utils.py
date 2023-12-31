import copy
import itertools
import numpy as np
import os
import scipy
import torch

from datetime import datetime


# particles: [B, seq_len, p, state_dim]
def compute_stats(args, particles):
    mean_p = torch.mean(particles[:, :, :args.n_particles], (1, 2))
    std_p = torch.mean(torch.std(particles[:, :, :args.n_particles], 2), 1)
    if args.surface_sample:
        mean_d = torch.mean(
            particles[:, 1:, :args.n_particles] - particles[:, :-1, :args.n_particles], (1, 2))
        std_d = torch.mean(torch.std(
            particles[:, 1:, :args.n_particles] - particles[:, :-1, :args.n_particles], 2), 1)
    else:
        d_list = []
        for b in range(particles.shape[0]):
            d_batch = []
            for j in range(particles.shape[1] - 1):
                x = particles[b, j+1, :args.n_particles]
                y = particles[b, j, :args.n_particles]
                x_ = x[:, None, :].repeat(1, y.size(0), 1)  # x: [N, M, D]
                y_ = y[None, :, :].repeat(x.size(0), 1, 1)  # y: [N, M, D]
                dis = torch.norm(torch.add(x_, -y_), 2, dim=2)  # dis: [N, M]
                cost_matrix = dis.cpu().numpy()
                ind1, ind2 = scipy.optimize.linear_sum_assignment(
                    cost_matrix, maximize=False)
                d_batch.append(torch.mean(x[ind1] - y[ind2], 0))

            d_list.append(torch.stack(d_batch))

        d_list = torch.stack(d_list)
        mean_d = torch.mean(d_list, 1)
        std_d = torch.std(d_list, 1)

    return mean_p, std_p, mean_d, std_d


# @profile
def find_relations_neighbor(kd_tree, pos, anchor_idx, radius, order, query_idx=None):
    '''
    The function first calls the query_ball_tree or query_ball_point method of the kd_tree object 
    to find the indices of the neighboring points for each anchor point or query point. 
    If query_idx is None, query_ball_tree is used to find the neighbors of the anchor points. 
    Otherwise, query_ball_point is used to find the neighbors of the query point(s).

    The function then constructs a 2D relations array of shape (neighbor_count_sum, 2) 
    where neighbor_count_sum is the sum of the number of neighboring points for all anchor points or query points. 
    The first column of relations contains the indices of the anchor points or query points, 
    and the second column contains the indices of the neighboring points.
    '''
    if query_idx is None:
        neighbor_ind = kd_tree.query_ball_tree(kd_tree, radius, p=order)
    else:
        neighbor_ind = kd_tree.query_ball_point(
            pos[query_idx], radius, p=order)

    neighbor_count_list = [len(i) for i in neighbor_ind]
    neighbor_count_sum = sum(neighbor_count_list)

    relations = np.zeros((neighbor_count_sum, 2), dtype=np.int16)

    if query_idx is None:
        relations[:, 0] = np.repeat(anchor_idx, neighbor_count_list)
        relations[:, 1] = anchor_idx[np.fromiter(
            itertools.chain.from_iterable(neighbor_ind), dtype=np.int16)]
    else:
        relations[:, 0] = anchor_idx[np.fromiter(
            itertools.chain.from_iterable(neighbor_ind), dtype=np.int16)]
        relations[:, 1] = np.repeat(query_idx, neighbor_count_list)

    return relations


def compute_neighbor_matrix(kd_tree, radius, order, other=None):
    if other is None:
        sps_dist_mat = kd_tree.sparse_distance_matrix(
            kd_tree, radius, p=order, output_type='coo_matrix')
    else:
        sps_dist_mat = kd_tree.sparse_distance_matrix(
            other, radius, p=order, output_type='coo_matrix')

    return sps_dist_mat.sign()


def prepare_input(args, state_cur, action_his=None, rels_list_prev=None, device='cpu'):
    B = state_cur.shape[0]
    n_nodes = state_cur.shape[1]
    tool_start_idx = n_nodes-args.gripper_dim

    # TODO:  args.n_particles includes the staic particles, which should be excluded?
    particle_ind = np.arange(args.n_particles)
    max_n_rel = 0
    rels_list_batch = []
    p2t_edge_idx_batch = []
    for k in range(B):
        state = state_cur[k]

        p2t_edge_idx_list = []
        particle_kd_tree = scipy.spatial.cKDTree(state[particle_ind])

        tool_idx = tool_start_idx
        tool_dim = np.sum(args.gripper_dim)
        tool_ind = np.arange(tool_idx, tool_idx + tool_dim)
        tool_kd_tree = scipy.spatial.cKDTree(state[tool_ind])
        sdm = particle_kd_tree.sparse_distance_matrix(
            tool_kd_tree, args.tool_neighbor_radius, output_type='ndarray')
            
        # t2p_dict map each point in tool_kd_tree to the indices of the points in particle_kd_tree
        t2p_dict = {}
        for j, (p, t, d) in enumerate(sdm):
            if t in t2p_dict:
                t2p_dict[t].append(j)
            else:
                t2p_dict[t] = [j]
                

        t2p_ind = list(itertools.chain.from_iterable(t2p_dict.values()))
        if len(t2p_ind) > 0:
            # p2t_edge_idx: pairs of (object idx, tool idx)
            p2t_edge_idx = np.array(
                [[x[0], x[1] + tool_idx] for x in sdm[t2p_ind]])
            # p2t_edge_idx_list: edge index for objects and the tool
            p2t_edge_idx_list.append(p2t_edge_idx)

        # for i in range(len(args.tool_dim)):
        #     tool_dim = args.tool_dim[i]

        #     tool_ind = np.arange(tool_idx, tool_idx + tool_dim)
        #     tool_kd_tree = scipy.spatial.cKDTree(state[tool_ind])
        #     sdm = kd_tree.sparse_distance_matrix(
        #         tool_kd_tree, args.tool_neighbor_radius[i], output_type='ndarray')

        #     t2p_dict = {}
        #     # TODO:  verify
        #     # object idx, tool idx, distance
        #     for j, (p, t, d) in enumerate(sdm):
        #         if t in t2p_dict:
        #             # if (len(t2p_dict[t]) < args.tool_neighbor_max[args.env][i]):
        #             t2p_dict[t].append(j)
        #             # else:
        #             #     if d < min(t2p_dict[t]):
        #             #         t2p_dict[t][np.argmax(t2p_dict[t])] = j
        #         else:
        #             t2p_dict[t] = [j]

        #     t2p_ind = list(itertools.chain.from_iterable(t2p_dict.values()))
        #     if len(t2p_ind) > 0:
        #         # p2t_edge_idx: pairs of (object idx, tool idx)
        #         p2t_edge_idx = np.array(
        #             [[x[0], x[1] + tool_idx] for x in sdm[t2p_ind]])
        #         # p2t_edge_idx_list: edge index for objects and two tools
        #         p2t_edge_idx_list.append(p2t_edge_idx)
        #     tool_idx += tool_dim

        if len(p2t_edge_idx_list) == 0:
            p2t_edge_idx_batch.append(np.zeros((0, 2), dtype=np.int16))
        else:
            p2t_edge_idx_batch.append(
                np.concatenate(p2t_edge_idx_list, axis=0))
            
        if getattr(args, 'tool_next_edge', False):
            tool_next = action_his[k]
            tool_next_kd_tree = scipy.spatial.cKDTree(tool_next)
            sdm = particle_kd_tree.sparse_distance_matrix(tool_next_kd_tree, args.tool_neighbor_radius, output_type='ndarray')
            t2p_dict = {}
            for j, (p, t, d) in enumerate(sdm):
                if t in t2p_dict:
                    t2p_dict[t].append(j)
                else:
                    t2p_dict[t] = [j]
            t2p_ind = list(itertools.chain.from_iterable(t2p_dict.values()))
            if len(t2p_ind) > 0:
                # p2t_edge_idx: pairs of (object idx, tool idx)
                p2t_edge_idx = np.array(
                    [[x[0], x[1] + tool_idx] for x in sdm[t2p_ind]])
                # p2t_edge_idx_list: edge index for objects and the tool
                p2t_edge_idx_list.append(p2t_edge_idx)


        if len(p2t_edge_idx_list) == 0 and rels_list_prev is not None:
            max_n_rel = max(max_n_rel, rels_list_prev[k][-1].shape[0])
            rels_list_batch.append(rels_list_prev[k][-1:])
        else:
            # I should only calculate the relations between static particles and dynamic particles
            if getattr(args, 'dynamic_staic_edge', False):
                p2p_edge_idx = find_relations_neighbor(
                    particle_kd_tree, state, particle_ind, args.neighbor_radius, 2, query_idx=np.arange(args.n_particles+args.floor_dim))
            else:
                p2p_edge_idx = find_relations_neighbor(
                    particle_kd_tree, state, particle_ind, args.neighbor_radius, 2 )

            # neighbors_ind = kd_tree.query_pairs(args.neighbor_radius, p=2, output_type='ndarray')
            # self_rels = np.stack((particle_ind, particle_ind)).T
            # p2p_edge_idx = np.concatenate((neighbors_ind, neighbors_ind[:, [1, 0]], self_rels), axis=0)
            p2t_edge_idx_list.append(p2p_edge_idx)
            max_n_rel = max(max_n_rel, sum(
                [x.shape[0] for x in p2t_edge_idx_list]))
            rels_list_batch.append(p2t_edge_idx_list)

    Rr_curs = torch.zeros((B, max_n_rel, n_nodes),
                          device=device, dtype=torch.float32)
    Rs_curs = torch.zeros((B, max_n_rel, n_nodes),
                          device=device, dtype=torch.float32)
    for k in range(B):
        p2t_edge_idx_list = np.concatenate(rels_list_batch[k], axis=0)
        rels_ind = torch.arange(
            p2t_edge_idx_list.shape[0], device=device, dtype=torch.int64)
        Rr_curs[k, rels_ind, p2t_edge_idx_list[:, 0]] = 1
        Rs_curs[k, rels_ind, p2t_edge_idx_list[:, 1]] = 1
    
    # Rr_curs and Rs_curs are the binary relation matrices for the current state
    # The binary relation matrices have a value of 1 in the (k, i, j) position if particle i is related to node j in the k-th batch, and 0 otherwise
    # 
    return Rr_curs, Rs_curs, rels_list_batch, p2t_edge_idx_batch
