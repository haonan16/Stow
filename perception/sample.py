import PIL
from matplotlib.pyplot import title
# import open3d as o3d as o3d
import cv2

import copy
import glob
import numpy as np
import os
import pdb
import sys
import imageio
from collections import OrderedDict
import functools
from scipy.optimize import minimize
import multiprocessing as mp

from datetime import datetime
from perception.pcd_utils import *
from pysdf import SDF
from timeit import default_timer as timer
from transforms3d.quaternions import *
from tqdm import tqdm
from config.config import gen_args
from utils.data_utils import *
from utils.visualize import *
from utils.utils3d import *
import utils.robot_utils as robot_utils
from planning.control_utils import *

from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_real_depth_map, get_camera_extrinsic_matrix, parse_intrinsics, get_xyz_from_depth
import robosuite.utils.macros as macros
import robosuite.utils.transform_utils as trans

import robosuite as suite

from skills.skill_controller import SkillController
# os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so"
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["MUJOCO_GL"] = "egl"
# os.environ["MUJOCO_PY_MJPRO_PATH"] = "/home/haonan/.mujoco/mujoco200"
# os.environ["MUJOCO_PY_EGL"] = "True"

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"


def get_individual_pcd(cam_name, obs, seg_id, cam_width, cam_height, crop_bound=None, visualize=False, env=None):
    # Load the RGB-D image and camera parameters
    fx, fy, cx, cy = parse_intrinsics(get_camera_intrinsic_matrix(
        env.sim, cam_name, cam_height, cam_width))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=cam_width, height=cam_height, fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsic = get_camera_extrinsic_matrix(env.sim, cam_name)

    binary_mask = (obs[f"{cam_name}_segmentation_element"]
                   [:, :, 0] == seg_id).astype(np.uint8)
    # Apply the segmentation mask to the RGB-D image
    seg_rgb = cv2.bitwise_and(
        obs[f"{cam_name}_image"], obs[f"{cam_name}_image"], mask=binary_mask)
    real_depth_map = get_real_depth_map(env.sim, obs[f"{cam_name}_depth"])
    seg_depth = cv2.bitwise_and(
        real_depth_map, real_depth_map, mask=binary_mask)

    rgb = o3d.geometry.Image((seg_rgb))
    depth = o3d.geometry.Image(seg_depth)

    # Convert the RGB-D image to a point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic)

    # Transform the point cloud to the world frame
    pcd.transform(extrinsic)

    # Define the bounding box
    if crop_bound is None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[-0.2, -0.8, 0.8], max_bound=[0.5, 0.32, 1.32])
    else:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=crop_bound[0], max_bound=crop_bound[1])

    if visualize:
        bbox.color = (1, 0, 0)
        o3d.visualization.draw_geometries([pcd, bbox])
    cropped_pcd = pcd.crop(bbox)

    return cropped_pcd


def get_pcd_from_rgbd(obs, camera_names, desired_seg_id, crop_bound=None, visualize=False, env=None):

    def f(x): return [get_individual_pcd(cam_name, obs, x, cam_width, cam_height, crop_bound, visualize, env=env)
                      for cam_name, cam_height, cam_width in zip(camera_names, env.camera_heights, env.camera_widths)]
    pcd_object_dict = OrderedDict(
        map(lambda kv: (kv[0], f(kv[1])), desired_seg_id.items()))

    return pcd_object_dict


def merge_pcds(pcd_list):
    return functools.reduce(lambda x, y: x + y, pcd_list)


def merge_point_cloud(args, camera_names, obs, desired_seg_id, crop_bound=None, visualize=False, env=None):

    pcd_object_dict = get_pcd_from_rgbd(
        obs, camera_names, desired_seg_id, crop_bound, visualize, env)

    pcd_object_dict = OrderedDict(
        map(lambda kv: (kv[0], merge_pcds(kv[1])), pcd_object_dict.items()))

    if visualize:
        visualize_o3d(list(pcd_object_dict.values()),
                      title='merged_and_cropped_raw_point_cloud')

    return pcd_object_dict


def preprocess_object_raw_pcd(args, pcd_object, pcd_all_objects, rm_stats_outliers=1, visualize=False, env=None):
    # Preprocessing: Voxel down-sampling
    pcd_object.voxel_down_sample(voxel_size=0.005)

    if rm_stats_outliers:
        rm_iter = 0
        outliers = None
        outlier_stat = None
        # remove until there's no new outlier
        # outlier_stat is None or len(outlier_stat.points) > 0:
        while rm_iter < 2:
            cl, inlier_ind_pcd_stat = pcd_object.remove_statistical_outlier(
                nb_neighbors=50, std_ratio=5.5+1.5*rm_iter)
            pcd_stat = pcd_object.select_by_index(inlier_ind_pcd_stat)
            outlier_stat = pcd_object.select_by_index(
                inlier_ind_pcd_stat, invert=True)
            if outliers is None:
                outliers = outlier_stat
            else:
                outliers += outlier_stat

            pcd_object = pcd_stat
            rm_iter += 1

            # press needs those points
            if 'press' in args.env or rm_stats_outliers == 1:
                break

    if visualize:
        outliers.paint_uniform_color([1, 0, 0.0])
        visualize_o3d([pcd_object, outliers], title='cleaned_workspace')

    selected_mesh = alpha_shape_mesh_reconstruct(
        pcd_object, alpha=0.005, mesh_fix=False, visualize=visualize)

    selected_surface = o3d.geometry.TriangleMesh.sample_points_poisson_disk(
        selected_mesh, args.n_particles)
    surface_points = np.asarray(selected_surface.points)

    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

    if visualize:
        visualize_o3d([surface_pcd], title='surface_point_cloud',
                      pcd_color=[0, 0.0, 1.0])
    sampled_pcd = surface_pcd

    # ##### 6. filter out the noise #####
    cl, inlier_ind_stat = sampled_pcd.remove_statistical_outlier(
        nb_neighbors=50, std_ratio=1.5)
    sampled_pcd_stat = sampled_pcd.select_by_index(inlier_ind_stat)
    outliers_stat = sampled_pcd.select_by_index(inlier_ind_stat, invert=True)
    sampled_pcd = sampled_pcd_stat
    outliers = outliers_stat

    if visualize:
        sampled_pcd.paint_uniform_color([0.0, 0.8, 0.0])
        outliers.paint_uniform_color([0.8, 0.0, 0.0])
        visualize_o3d([pcd_object, sampled_pcd, outliers],
                      title='cleaned_point_cloud', pcd_color=[0, 0.0, 1.0])

    try:
        obb = sampled_pcd.get_minimal_oriented_bounding_box()
    except:
        obb = sampled_pcd.get_oriented_bounding_box()
    if visualize:
        obb.color = (1, 0, 0)
        visualize_o3d([pcd_object, sampled_pcd, obb],
                      title='cleaned_point_cloud', pcd_color=[0, 0.0, 1.0])

    # Get the box dimensions
    dim = obb.extent

    # Create a cuboid mesh
    cube_mesh = o3d.geometry.TriangleMesh.create_box(
        width=dim[0], height=dim[1], depth=dim[2])

    # Scale and transform the mesh using the OBB parameters
    # mesh.scale(obb.extent)
    cube_mesh.rotate(R=obb.R, center=(0, 0, 0))
    translation = np.asarray(obb.center)
    cube_mesh.translate(translation)

    if visualize:
        visualize_o3d([cube_mesh, obb],
                      title='cleaned_point_cloud', pcd_color=[0, 0.0, 1.0])

    return sampled_pcd




def pcd_dis_wall(args, pcd_points, wall, dim, visualize=False):
    """
    Compute the distance between a point cloud and a wall
    Args:
        pcd_points:  number of points * 3
    return: 
        pcd_dis:  number of points * 1
    """
    wall = np.array(wall)[None,]
    diff = pcd_points[:args.n_particles, dim] - wall[:, dim]
    distance = diff * np.sign(diff)
    
    # Calculate the difference in shape
    diff_shape = int(np.array(pcd_points.shape[0]) - np.array(distance.shape))

    # Pad array "distance" with zeros to match the shape of array "pcd_points"
    distance_padded = np.pad(distance, (0, diff_shape), mode='constant')

    return distance_padded

def create_static_pcd(args, env_config):
    static_pcd = OrderedDict()
    num_particles = args.n_particles_per_object
    # shelf height is env.shelf_pos[2]-0.38
    static_pcd['table'] = generate_box_point_cloud(
        args, 
        env_config['table']['pos'], 
        env_config['table']['quat'], 
        env_config['table']['size'], 
        np_color=np.array([1, 1, 0]), 
        n_particles=num_particles[f'table'], static=True, visualize=False)

    shelf_pcd = OrderedDict()
    shelf_bottom_size = env_config['shelf_bottom']['size'].copy()
    shelf_bottom_size[1]=0.4
    shelf_pcd['shelf_bottom'] = generate_box_point_cloud(
        args,
        env_config['shelf_bottom']['pos'], 
        env_config['shelf_bottom']['quat'], 
        shelf_bottom_size,            
        np_color=np.array([0, 1, 1]),
        n_particles=num_particles[f'shelf_bottom'],
        static=True, visualize=False)

    shelf_pcd['shelf_right'] = generate_box_point_cloud(
        args,
        env_config['shelf_right']['pos'], 
        env_config['shelf_right']['quat'], 
        env_config['shelf_right']['size'],            
        np_color=np.array([0, 1, 1]),
        n_particles=num_particles[f'shelf_right'],
        static=True, visualize=False)

    shelf_pcd['shelf_left'] = generate_box_point_cloud(
        args,
        env_config['shelf_left']['pos'], 
        env_config['shelf_left']['quat'], 
        env_config['shelf_left']['size'],            
        np_color=np.array([0, 1, 1]),
        n_particles=num_particles[f'shelf_left'],
        static=True, visualize=False)
    
    if args.evenly_spaced:
        static_pcd['shelf'] = np.concatenate(list(shelf_pcd.values()), axis=0)
    else:
        static_pcd['shelf'] = functools.reduce(
            lambda x, y: x+y, shelf_pcd.values())
    return static_pcd

def preprocess_raw_pcd(args, obs, static_pcd, env_config, camera_names=['agentview'], rm_stats_outliers=1, visualize=False, env=None):
    # Set the desired segmentation label
    desired_seg_id = OrderedDict([('cubeA', np.int32(66)),
                                  ('cubeB_0', np.int32(68)),
                                  ('cubeB_1', np.int32(70)),
                                  ('table', np.int32(8)),
                                  ('shelf_bottom', np.int32(14)),
                                  ('shelf_right', np.int32(16)),
                                  ('shelf_left',  np.int32(18)),
                                  ])
    num_particles = args.n_particles_per_object
    if args.rescontruct_from_pose:
        selected_pcd_objects = OrderedDict()
        
        ################################ pcd for dynamic objects ################################
        selected_pcd_objects['cubeA'] = generate_box_point_cloud(
            args, obs['cubeA_pos'], obs['cubeA_quat'], np.array(env_config['cubeA_size'])*2, np_color=np.array([1, 0, 0]), n_particles=num_particles['cubeA'], visualize=False)

        cube_color = np.array([[0, 1, 0],
                              [1, 0, 1]])
        shelf_obj_pcd = OrderedDict([(f'cubeB_{i}', generate_box_point_cloud(
            args, obs[f'cubeB_{i}_pos'], obs[f'cubeB_{i}_quat'], np.array(env_config[f'cubeB_{i}_size'])*2, np_color=[0, 1, 0], n_particles=num_particles[f'cubeB_{i}'],  visualize=False)) 
                                    for i in np.arange(env_config.get('num_obj_on_shelf'))])
        selected_pcd_objects.update(shelf_obj_pcd)
        for i in np.arange(args.max_shelf_objs):
            if f'cubeB_{i}' not in selected_pcd_objects.keys():
                if args.evenly_spaced:
                    selected_pcd_objects[f'cubeB_{i}'] = args.dummy_box_state
                else:
                    selected_pcd_objects[f'cubeB_{i}'] = None

        ################################ pcd for static objects ################################
        
        selected_pcd_objects.update(static_pcd)

        ################################ pcd for gripper ################################
        gripper_width = robot_utils.get_gripper_width(obs['robot0_gripper_qpos'])
        selected_pcd_objects['gripper'] = create_gripper_pcd(args, obs['robot0_eef_pos'], obs['robot0_eef_quat'], gripper_width=gripper_width,
                                                             gripper_color=[0, 0, 1], n_particles=num_particles['gripper'], visualize=False)

        # # Define the bounding box
        # bbox = o3d.geometry.AxisAlignedBoundingBox(
        #     min_bound=crop_bound[0], max_bound=crop_bound[1])

        # # Applying a crop function to all values in the ordered dictionary
        # selected_pcd_objects = OrderedDict((key, value.crop(bbox)) for key, value in selected_pcd_objects.items())

        if visualize:
            # o3d.visualization.draw_geometries(list(selected_pcd_objects.values()))
            visualize_o3d(args, list(selected_pcd_objects.values()),
                          title='processed_pcd')
            # np.array(list(len(i.points) for i in selected_pcd_objects.values()))
        if args.evenly_spaced:
            pcd_pts = np.concatenate([points  for points in selected_pcd_objects.values()], axis=0)
        else:
            pcd_pts = np.concatenate([np.asarray(pcd.points) if pcd !=None else args.dummy_box_state
                                    for pcd in selected_pcd_objects.values() ], axis=0)
        
        wall_pos_y = [wall_pos[1] for wall_pos in env_config['wall_pos'].values()]
        # dis_wall = []
        # for wall_pos in env_config['wall_pos'].values():
        #     dis_to_wall = pcd_dis_wall(args, pcd_pts, wall_pos, dim=1)
        #     dis_wall.append(dis_to_wall[..., None])
        # # Combine the three arrays into a new array with shape (3, 3, 4)
        # dis_wall = np.concatenate(dis_wall, axis=1)

        if args.evenly_spaced:
            obj_gt_state, obj_id, attr = zip(*[get_single_obj_state(obj_name, obj_state, id,
                                            args, args.max_n_instance) for id, (obj_name, obj_state) in enumerate(selected_pcd_objects.items())])

        else:
            obj_gt_state, obj_id, attr = zip(*[get_single_obj_state(obj_name, obj_state, id,
                                            args, args.max_n_instance) for id, (obj_name, obj_state) in enumerate(selected_pcd_objects.items())])
        stored_obs = {k: v for k, v in obs.items() if 'image' not in k and 'depth' not in k and 'seg' not in k }

        h5_data = [np.vstack(obj_gt_state), np.vstack(obj_id), np.vstack(attr), wall_pos_y, stored_obs]

    else:
        pcd_object_dict = merge_point_cloud(
            args, camera_names, obs, desired_seg_id, crop_bound=None, visualize=False, env=env)

        selected_pcd_objects = OrderedDict(
            map(lambda kv: (kv[0], preprocess_object_raw_pcd(
                args, kv[1], list(pcd_object_dict.values()), rm_stats_outliers, visualize, env)), pcd_object_dict.items()))

    return h5_data


def save_camera_mat(env, rollout_dir, i):

    os.makedirs(f"{rollout_dir}/{i:03d}", exist_ok=True)
    cam_params = {}

    with open(f"{rollout_dir}/{i:03d}"+"/cam_params.npy", 'wb') as f:
        for cam_name, cam_height, cam_width in zip(env.camera_names, env.camera_heights, env.camera_widths):
            cam_params['{}_ext'.format(cam_name)] = get_camera_extrinsic_matrix(
                env.sim, cam_name)
            intrinsic = get_camera_intrinsic_matrix(
                env.sim, cam_name, cam_height, cam_width)
            if 'intrinsic' not in cam_params:
                cam_params['intrinsic'] = intrinsic
            assert np.array_equal(
                intrinsic, cam_params['intrinsic']), "intrinsic matrix is not the same for all cameras"
        np.save(f, cam_params)


def get_single_obj_state(obj_name, obj_state, id, args, max_n_instance):
    '''helper function to get the state of a single object'''
    if obj_state is None:
        obj_gt_state = args.dummy_box_state
    else:
        if args.evenly_spaced:
            obj_gt_state = obj_state
        else:
            obj_gt_state = np.asarray(obj_state.points)
    obj_id = np.zeros((obj_gt_state.shape[0], max_n_instance))
    obj_id[:, id] = 1
    attr = np.zeros((obj_gt_state.shape[0], args.attr_dim))
    # attribute 0: [movable, fixed]
    if 'cube' in obj_name or 'gripper' in obj_name:
        attr[..., 0] = 1  # movable
    else:
        attr[..., 0] = 0  # fixed
    if np.array_equal(obj_gt_state, args.dummy_box_state) :
        attr[..., 1] = 0 # dummy
    else:
        attr[..., 1] = 1 # valid
        
    return obj_gt_state, obj_id, attr


def save_data_step(args, h5_data, obs, rollout_dir, rollout_idx, skill_idx, camera_names=None, verbose=False):
    if camera_names != None:
        for cam_name in camera_names:
            cv2.imwrite(f"{rollout_dir}/{rollout_idx:03d}/{skill_idx:03d}_rgb_{cam_name}.png",
                        obs[f"{cam_name}_image"][..., ::-1])
    if verbose:
        render_processed_pcd(args, h5_data[0], obs, path=f"{rollout_dir}/{rollout_idx:03d}/{skill_idx:03d}.png", camera_names=camera_names)
    # if 'agentview_image' in obs:
    #     cv2.imwrite(f"{rollout_dir}/{rollout_idx:03d}/{skill_idx:03d}_rgb_agentview.png",
    #                 obs[f"agentview_image"][..., ::-1])

    # with open(f"{rollout_dir}/{rollout_idx:03d}/{skill_idx:03d}_pcd.npy", 'wb') as f:
    #     np.save(f, OrderedDict(map(lambda kv: (kv[0], np.asarray(kv[1].points)), pcd_dict.items())))

    store_data(args.data_names, h5_data,
               f"{rollout_dir}/{rollout_idx:03d}/{skill_idx:03d}.h5")
    
    # cv2.imwrite(f"{rollout_dir}/{rollout_idx:03d}/{skill_idx:03d}_pcd.png", cv2.cvtColor(point_cloud_to_image(list(pcd_dict.values())), cv2.COLOR_RGB2BGR))


def obj_within_box(crop_bound, obs, env_config):
    cubeB_within_box = all([point_within_box(obs[f"cubeB_{i}_pos"], box_min=np.array(crop_bound[0]), box_max=np.array(crop_bound[1])) for i in range(env_config['num_obj_on_shelf'])])
    cubeA_within_box = point_within_box(obs[f"cubeA_pos"], box_min=np.array(crop_bound[0]), box_max=np.array(crop_bound[1]))
    return cubeB_within_box and cubeA_within_box


def gen_rollouts(rollout_infos):
    thread_idx = rollout_infos['thread_idx']
    rollout_dir = rollout_infos['rollout_dir']
    max_skill_calls = rollout_infos['max_skill_calls']
    n_rollouts = rollout_infos['n_rollouts']
    args = rollout_infos['args']

    # create environment for data processing
    env = suite.make(
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        args=args,
        ignore_done=True,
        control_freq=20,
        env_name='Stow'+args.skill_name.capitalize(),
        **args.sim_kwargs,
        )
    set_seed(args.dataset_seed+thread_idx)
    verbose = True
    skill_controller = SkillController(args, args.skill_config)
    skill_controller.reset_to_skill(args.skill_name)
    # i = 0
    zero_action = np.array(
        [-0.027, -0.048, 0.964, -2.22110601,  2.22110601,  0., -1])
    
    invalid_episode = False
    for i in tqdm(range(n_rollouts), desc="Generating rollouts", total=n_rollouts, leave=True):
        rollout_idx = thread_idx * n_rollouts + i

    # while i < n_rollouts:
        # print(f"+++++++++++++++++++Rollout: {i}+++++++++++++++++++++")
        obs = env.reset()
        for _ in range(3):
            obs, _, _, _ = env.step(zero_action)

        os.system('mkdir -p ' + f"{rollout_dir}/{rollout_idx:03d}")

        # create a video writer with imageio
        save_camera_mat(env, rollout_dir, rollout_idx)
        
        # save the model xml
        xml_path = os.path.join(f"{rollout_dir}/{rollout_idx:03d}", "model.xml")
        with open(xml_path, "w") as f:
            f.write(env.sim.model.get_xml())
        env_config = env.env_config
        store_json(env_config, os.path.join(f"{rollout_dir}/{rollout_idx:03d}", "env_config.json"))
        
        static_pcd = create_static_pcd(args, env.env_config)
        if verbose:
            video_writer = imageio.get_writer(
                f'{rollout_dir}/{rollout_idx:03d}/repr.mp4', fps=20)

        frame_idx = 0
        behavior_idx = 0
        for skill_idx in np.arange(max_skill_calls):

            hl_param = skill_controller.random_hl_param(obs, env_config)
            # keyposes = skill_controller._cur_skill.get_keyposes()

            if args.debug:
                print(
                    f'Sample new parameters for behavior primitives at frame_idx: {frame_idx}')

            while True:
                frame_idx += 1
                ll_action = skill_controller.step_ll_action(obs)
                # ll_action = np.array([-0.0682756 , -0.08587694 , 0.95 ,      -2.22110601  ,2.22110601 , 0., -1.        ])
                next_obs, _, done, _ = env.step(ll_action)
                if args.debug:
                    print('ll_action:', ll_action, ' robot_pos:',
                          next_obs['robot0_eef_pos'])
                if verbose:
                    video_writer.append_data(next_obs[f"paper_image"])
                obs = copy.deepcopy(next_obs)
                # print(f"frame_idx: {frame_idx}")
                # update for next iter
                if not obj_within_box(args.crop_bound, obs, env_config):
                    invalid_episode = True
                    if verbose:
                        video_writer.close()

                    with open(os.path.join(f"{rollout_dir}/{rollout_idx:03d}", "invalid_sample.txt"), "w"):
                        pass
                    break

                if skill_controller.done():
                    # print("finsihsed one skill")
                    break
                
                expected_cur_pose = skill_controller.is_keyframe_reached()
                if expected_cur_pose != None:                    
                    tool_repr = keypose_to_tool_repr(args, expected_cur_pose, obs)
                    h5_data = preprocess_raw_pcd(
                        args, obs, static_pcd, env_config, env.camera_names, env=env)
                    h5_data.append(tool_repr)
                    save_data_step(args, h5_data, obs,
                                   rollout_dir, rollout_idx, behavior_idx, env.camera_names)
                    behavior_idx += 1

            if invalid_episode:
                invalid_episode = False
                break

            behavior_idx += 1
        if verbose:
            video_writer.close()

        # i += 1
        # Update the progress bar using tqdm
        # tqdm.write(f"Rollout Progress: {i}/{n_rollouts}")



def main():
    args = gen_args()


    task_name = 'Stow_'

    time_now = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    rollout_dir = os.path.join(
        cd, '..', f'dump/perception/{task_name}{args.skill_name}')
    print(f'rollout_dir: {rollout_dir}')

    if args.gen_rollout:
        rollout_infos = []
        rollout_dir = os.path.join(rollout_dir, time_now)
        args.num_workers = max(1, args.num_workers)
        for i in range(args.num_workers):
            rollout_info = {'thread_idx': i,
                            'rollout_dir': rollout_dir,
                            'max_skill_calls': args.max_skill_calls,
                            'n_rollouts': args.n_rollouts//(args.num_workers),
                            'args': args}  # TODO: add random seed

            rollout_infos.append(rollout_info)
        if args.num_workers == 1:
            # # Instantiate the LineProfiler
            # lp = LineProfiler()
            # # Add functions to the profiler
            # lp.add_function(gen_rollouts)
            # # Enable the profiler
            # lp.enable_by_count()

            # Call your function with the desired arguments
            gen_rollouts(rollout_info)

            # # Disable the profiler
            # lp.disable_by_count()

            # # Print the profiling results
            # lp.print_stats()
            # # gen_rollouts(rollout_info)
        else:
            cores = args.num_workers
            pool = mp.Pool(processes=cores)
            pool.map(gen_rollouts, rollout_infos)


if __name__ == '__main__':
    main()
