import os
import numpy as np
import torch
import torch.nn.functional as F
import pdb

torch.set_default_tensor_type(torch.FloatTensor)

from datetime import datetime
from dynamics.dy_utils import *
from dynamics.model import Model, ChamferLoss, EarthMoverLoss
from dynamics.dataset import GNNDataset
from dynamics.eval import evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from config.config import *
from utils.data_utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# @profile
def train(args, model):
    # load training data
    if args.valid:
        phases = ['train', 'valid']
    else:
        phases = ['train']

    datasets = {phase: GNNDataset(args, phase) for phase in phases}

    # for phase in phases:
    #     datasets[phase].load_stats()
    dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers) for phase in phases}

    # configure optimizer
    if args.stage == 'dy':
        params = model.dynamics_predictor.parameters()
    else:
        raise AssertionError(f"Unknown stage: {args.stage}")

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    else:
        raise AssertionError(f"Unknown optimizer: {args.optimizer}")

    # reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)

    # define loss
    chamfer_loss = ChamferLoss(args.loss_ord)
    emd_loss = EarthMoverLoss(args.loss_ord)
    # dcd_loss = DCDLoss(args.dcd_alpha, args.dcd_n_lambda)

    # start training
    start_epoch = 0
    if len(args.resume_path) > 0:
        resume_name_list = os.path.basename(args.resume_path).split('_')
        for i in range(len(resume_name_list)):
            if 'epoch' in resume_name_list[i]:
                start_epoch = int(resume_name_list[i+1])
                break

    best_valid_loss, best_epoch = np.inf, 0
    plateau_epoch = 0

    train_stats = {
        'train_loss': [], 
        'valid_loss': [],
        'stats': []
    }

    mean_p = torch.tensor(args.mean_p, device=args.device, dtype=torch.float32)
    std_p = torch.tensor(args.std_p, device=args.device, dtype=torch.float32)
    mean_d = torch.tensor(args.mean_d, device=args.device, dtype=torch.float32) 
    std_d = torch.tensor(args.std_d, device=args.device, dtype=torch.float32)

    best_model_path = None
    last_model_path = None
    for epoch in range(start_epoch, start_epoch + args.n_epoch):
        for phase in phases:
            model.train(phase=='train')
            # meter_loss = AverageMeter()
            loss_list = []
            n_iters = len(dataloaders[phase])
            # each "data" is a trajectory of sequence_length time steps
            for i, data in enumerate(tqdm(dataloaders[phase], desc=f'Epoch {epoch}/{start_epoch + args.n_epoch}')):
                state_seq, obj_seq, attr_seq, tool_seq = data
                
                B = state_seq.size(0)
                n_nodes = state_seq.size(2)
                
                # p_rigid: B x n_instance
                # physics_param: B x n_param
                memory_init = model.init_memory(B, n_nodes) # [p_rigid, physics_param]
                groups_gt = get_env_group(args, B)
                # groups_gt.insert(1, obj_seq)

                # floor_state =  torch.tile(torch.FloatTensor(args.floor_state), (B, 1, 1)).to(args.device)

                if args.batch_norm:
                    mean_p, std_p, _, _ = compute_stats(args, state_seq)
                
                stats = [mean_p, std_p, mean_d, std_d]
                train_stats['stats'].append(stats)
                
                tool_start_idx = n_nodes - args.gripper_dim
                loss = 0
                for j in range(args.sequence_length): # the number of timestep to predict
                    with torch.set_grad_enabled(phase=='train'):
                        # action_his.shape: B x 2(before and after) x tool_size x 3
                        # action_his = state_seq[:, args.n_his*args.time_gap*j:args.n_his*args.time_gap*(j+1)+1,tool_start_idx: ]
                        action_his = tool_seq

                        if j == 0:
                            state_cur = state_seq[:, args.n_his-1, :args.n_particles+args.floor_dim]
                        else:
                            state_cur = pred_pos_p
                        state_cur = torch.cat([state_cur, action_his[:, -1]], dim=1)

                        Rr_curs, Rs_curs, _, _ = prepare_input(args, state_cur[:, :, :3].detach().cpu().numpy(), action_his[:,1].detach().cpu().numpy(), device=args.device)
                        inputs = [attr_seq, state_cur, action_his, Rr_curs, Rs_curs, memory_init, groups_gt, stats]
                        # pred_pos_p (unnormalized): B x n_p x state_dim
                        # pred_motion_p (normalized): B x n_p x state_dim
                        pred_pos_p, pred_motion_p, _, pred_tool_p, pred_transformations = model.predict_dynamics(inputs)
                        # concatenate the state of the shapes
                        # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
                        # pred_pos = torch.cat([pred_pos_p, state_seq[:, args.n_his+j, args.n_particles:]], 1)

                        gt_pos_p = state_seq[:, args.n_his+j, :args.n_particles_type[0], :3]
                        gt_tool_p = state_seq[:, args.n_his+j, tool_start_idx:, :3]
                        # gt_motion_p = gt_pos_p - state_seq[:, args.n_his+j-1, :args.n_particles]
                        # gt_motion_p_norm = batch_normalize(gt_motion_p, mean_d, std_d)

                        if args.loss_type == 'chamfer_emd':
                            if args.chamfer_weight > 0:
                                loss += args.chamfer_weight * (chamfer_loss(pred_pos_p, gt_pos_p)+chamfer_loss(pred_tool_p, gt_tool_p)) if args.auxiliary_gripper_loss else chamfer_loss(pred_pos_p, gt_pos_p)
                            if args.emd_weight > 0:
                                loss += args.emd_weight * (emd_loss(pred_pos_p, gt_pos_p)+emd_loss(pred_tool_p, gt_tool_p)) if args.auxiliary_gripper_loss else emd_loss(pred_pos_p, gt_pos_p)
                        elif args.loss_type == 'mse':
                            loss += F.mse_loss(pred_pos_p, gt_pos_p)+F.mse_loss(pred_tool_p, gt_tool_p) if args.auxiliary_gripper_loss else F.mse_loss(pred_pos_p, gt_pos_p)
                        elif args.loss_type == 'emd':
                            loss += emd_loss(pred_pos_p, gt_pos_p)+emd_loss(pred_tool_p, gt_tool_p) if args.auxiliary_gripper_loss else emd_loss(pred_pos_p, gt_pos_p)
                        # elif args.loss_type == 'mse_motion':
                        #     loss += F.mse_loss(pred_motion_p , gt_motion_p)
                        elif args.loss_type == 'l1':
                            loss += F.l1_loss(pred_pos_p, gt_pos_p)
                        else:
                            raise NotImplementedError

                # update model parameters
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i > 0 and ((epoch * n_iters) + i) % args.ckp_per_iter == 0:
                        last_model_path = f'{args.dy_out_path}/net_epoch_{epoch}_iter_{i}.pth'
                        torch.save(model.state_dict(), last_model_path)

                loss_list.append(loss)

            loss_avg = torch.mean(torch.stack(loss_list))
            print(f'{phase} epoch[{epoch}/{args.n_epoch}] lr: {round(get_lr(optimizer), 6)}, '
                + f'{phase} loss: {round(loss_avg.item(), 6)}')
            train_stats[f'{phase}_loss'].append(loss_avg.item())

            if phase == 'valid':
                scheduler.step(loss_avg)
                if loss_avg.item() < best_valid_loss:
                    best_valid_loss = loss_avg.item()
                    best_epoch = epoch
                    best_model_path = f'{args.dy_out_path}/net_best.pth'
                    torch.save(model.state_dict(), best_model_path)
                    plateau_epoch = 0
                else:
                    plateau_epoch += 1

        if plateau_epoch >= args.plateau_epoch_size:
            print(f'Breaks after not improving for {plateau_epoch} epoches!')
            break

    print(f"Best valid loss {round(best_valid_loss, 6)} is achieved at epoch {best_epoch}!")
    
    with open(os.path.join(args.dy_out_path, 'train_stats.npy'),'wb') as f:
        np.save(f, train_stats)

    if args.eval:
        if best_model_path is not None:
            args.dy_model_path = best_model_path
        elif last_model_path is not None:
            args.dy_model_path = last_model_path
        else:
            raise NotADirectoryError

    with open(os.path.join(args.dy_out_path, 'args.npy'), 'wb') as f:
        np.save(f, args.__dict__)
    store_json(args.__dict__, os.path.join(args.dy_out_path, 'args.json'))
    
    if args.eval:
        evaluate(args)


def get_test_name(args):
    test_name = ['dy']
    test_name.append(args.data_type)
    # test_name.append(f'p={args.n_particles}')
    # test_name.append(f'nr={args.neighbor_radius}')
    # test_name.append(f'tnr={args.tool_neighbor_radius}')

    test_name.append(f'tr={args.train_set_ratio}')

    # test_name.append(f'his={args.n_his}')
    # test_name.append(f'seq={args.sequence_length}')
    # test_name.append(f'time_gap={args.time_gap}')
    test_name.append(args.loss_type)

    # if args.loss_type == 'chamfer_emd':
    #     test_name.append(str(args.chamfer_weight))
    #     test_name.append(str(args.emd_weight))
    # if args.loss_type == 'dcd':
    #     test_name.append(str(args.dcd_alpha))
    #     test_name.append(str(args.dcd_n_lambda))
    if args.dynamic_staic_edge: test_name.append('dse')
    if args.auxiliary_gripper_loss: test_name.append('auxgp')

    if args.batch_norm: test_name.append(f'bn={args.batch_norm}')
    # if args.rigid_motion: test_name.append(f'rm={args.rigid_motion}')
    # if args.attn: test_name.append(f'attn={args.attn}')
    # if args.valid: test_name.append('valid')

    # if args.debug: test_name.append('debug')

    test_name.append(datetime.now().strftime("%b-%d-%H:%M:%S"))

    return '_'.join(test_name)


def main():
    args = gen_args()
    set_seed(args.random_seed)

    args.dy_out_path = os.path.join(args.dy_out_path, get_test_name(args))

    os.system('mkdir -p ' + args.dy_data_path)
    os.system('mkdir -p ' + args.dy_out_path)

    tee = Tee(os.path.join(args.dy_out_path, 'train.txt'), 'w')

    # create model and train
    model = Model(args)
    print(f"Model #params: {count_parameters(model)}")

    # resume training of a saved model (if given)
    if len(args.resume_path) > 0:
        print(f"Loading saved ckp from {args.resume_path}")
        if args.stage == 'dy':
            pretrained_dict = torch.load(args.resume_path)
            model_dict = model.state_dict()

            # only load parameters in dynamics_predictor
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() \
                if 'dynamics_predictor' in k and k in model_dict
            }
            model.load_state_dict(pretrained_dict, strict=False)

    if args.debug:
        args.num_workers = 0
        args.n_epoch = 1
        args.ckp_per_iter = 1000
        args.n_rollout = 2

    # log args
    print_args(args)

    model = model.to(args.device)
    train(args, model)


if __name__ == '__main__':
    main()
