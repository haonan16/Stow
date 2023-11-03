import copy
import cv2 as cv
import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np
import open3d as o3d 
import os
import pickle
import pymeshfix
import seaborn as sns
# sns.set_theme(style="darkgrid")
import sys
import torch
import torchvision.transforms as transforms
from vispy import app, gloo, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Markers
from vispy.visuals.transforms import STTransform

from datetime import datetime
from sklearn import metrics
from trimesh import PointCloud
import textwrap
from mpl_toolkits.mplot3d.art3d import Line3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.data_utils import *
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
from utils.data_utils import load_json
from scipy.spatial.transform import Rotation as R
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import robosuite.utils.transform_utils as trans

matplotlib.rcParams["legend.loc"] = 'lower right'
color_list = ['royalblue', 'red', 'green',
              'cyan', 'orange', 'pink', 'tomato', 'violet']


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def plot_train_loss(loss_dict, path=''):
    plt.figure(figsize=[16, 9])

    for label, loss in loss_dict.items():
        if not 'loss' in label and not 'accuracy' in label:
            continue
        time_list = list(range(len(loss)))
        plt.plot(time_list, loss, linewidth=6, label=label)

    plt.xlabel('epoches', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Training Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def plot_eval_loss(title, loss_dict, loss_std_dict=None, alpha_fill=0.3, colors=None,
                   path='', xlabel='Time / Steps', ylabel='Loss'):
    plt.figure(figsize=[16, 9])

    if not colors:
        colors = color_list

    i = 0
    for label, loss in loss_dict.items():
        time_list = list(range(len(loss)))
        # plt.plot(time_list, loss, linewidth=6, label=label, color=colors[i % len(colors)])

        # , color=colors[i % len(colors)])
        sns.lineplot(x=time_list, y=loss, label=label)

        # plt.annotate(str(round(loss[0], 4)), xy=(0, loss[0]), xytext=(-30, 20), textcoords="offset points", fontsize=20)
        # plt.annotate(str(round(loss[-1], 4)), xy=(len(loss)-1, loss[-1]), xytext=(-30, 20), textcoords="offset points", fontsize=20)

        if loss_std_dict:
            loss_min_bound = loss - loss_std_dict[label]
            loss_max_bound = loss + loss_std_dict[label]
            plt.fill_between(time_list, loss_max_bound, loss_min_bound,
                             color=colors[i % len(colors)], alpha=alpha_fill)

        i += 1

    plt.xlabel(xlabel, fontsize=30)
    plt.ylabel(ylabel, fontsize=30)
    plt.title(title, fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_points(ax, args, particles, target, axis_off=False, mask=None, rels=None, focus=True, res='high', env_config=None):
    if axis_off:
        ax.axis('off')

    color_map = plt.cm.Set1
    point_size = 4 if res == 'high' else 10
    outputs = []
    particles_idx = 0
    i = 0
    
    for obj_name, n_particles_per_object in args.n_particles_per_object_clean.items():
        if 'cube' in obj_name:
            color = color_map(i+1)
            i += 1
        elif 'gripper' in obj_name:
            color = color_map(0)
        # elif 'table' in obj_name or 'shelf' in obj_name:
        #     color = 'b'
        else:
            color = color_map(8)
        
        # Check if the point is not at (0,0,0) before plotting
        valid_indices = (particles[particles_idx:particles_idx + n_particles_per_object] != [0, 0, 0]).any(axis=1)
        valid_particles = particles[particles_idx:particles_idx + n_particles_per_object][valid_indices]

            
        points = ax.scatter(
            valid_particles[:, args.axes[0]],
            valid_particles[:, args.axes[1]],
            valid_particles[:, args.axes[2]],
            color=color, s=point_size)
        particles_idx = particles_idx + n_particles_per_object
        
        outputs.append(points)
        
        if rels is not None:
            for i in range(rels.shape[0]):
                if rels[i][0] > 0 and rels[i][1] > 0:
                    neighbor = ax.plot(particles[rels[i], args.axes[0]], particles[rels[i], args.axes[1]], 
                        particles[rels[i], args.axes[2]], c='g')
                    # neighbor = Line3D(particles[rels[i], args.axes[0]], particles[rels[i], args.axes[1]], 
                    #     particles[rels[i], args.axes[2]], c='g', linestyle='-', linewidth=0.5)

                else:
                    neighbor = ax.plot([], [], [], c='g')
                    # neighbor = Line3D([], [], [], c='g', linestyle='-', linewidth=0.5)

                outputs.append(neighbor[0])
                # outputs.append(neighbor)

    if target is not None:
        ax.scatter(target[:, args.axes[0]], target[:, args.axes[1]], target[:, args.axes[2]], c='y', s=point_size, alpha=0.3)

    centers = np.mean(np.array(particles[:args.n_particles]), axis=0)
    centers = [centers[args.axes[0]], centers[args.axes[1]], centers[args.axes[2]]]
    if focus:
        r = args.plot_radius# (int(np.sqrt(args.floor_dim)) - 1) * args.plot_scale / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    return outputs

def draw_box(ax, center, quat, size, color):
    rotation = R.from_quat(quat)
    rotation_matrix = rotation.as_matrix()

    box = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, size)
    vertices = np.asarray(box.get_box_points())
    edges = [[0, 1], [0, 3], [0, 4], [2, 1], [2, 6], [2, 3], [5, 1], [5, 4], [5, 6], [7, 3], [7, 4], [7, 6]]

    faces = Poly3DCollection([vertices[edge] for edge in edges], linewidths=1, edgecolors=color, alpha=0.1)
    faces.set_facecolor(color)

    ax.add_collection3d(faces)
    return faces

def visualize_boxes(ax, args, particles, target, axis_off=False, mask=None, rels=None, focus=True, res='high', env_config=None):
    from utils.utils3d import recover_box_properties

    if axis_off:
        ax.axis('off')

    color_map = plt.cm.Set1
    point_size = 4 if res == 'high' else 10
    outputs = []
    particles_idx = 0
    i = 0

    ax.clear()
    
    # Initialize an empty list to store the 3D patches
    box_patches = []
    
    for obj_name, n_particles_per_object in args.n_particles_per_object.items():
        # Check if the point is not at (0,0,0) before plotting
        if 'cubeB' in obj_name and int(obj_name[-1]) >= env_config['num_obj_on_shelf']:
            continue

        valid_indices = (particles[particles_idx:particles_idx + n_particles_per_object] != [0, 0, 0]).any(axis=1)
        valid_particles = particles[particles_idx:particles_idx + n_particles_per_object][valid_indices]
        particles_idx = particles_idx + n_particles_per_object
        if not valid_particles.any() :
            continue

        if 'cube' in obj_name:
            color = color_map(i+1)
            i += 1
            box_size = env_config[f'{obj_name}_size']

        elif 'gripper' in obj_name:
            color = color_map(0)
            box_size = args.gripper_size
        # elif 'table' in obj_name or 'shelf' in obj_name:
        #     color = 'b'
        else:
            color = color_map(8)
            box_size = env_config[f'{obj_name}']['size']
        

        center, quat = recover_box_properties(valid_particles)

        # Draw box for valid particles
        box_patch = draw_box(ax, center, quat, box_size, color)
        box_patches.append(box_patch)



    centers = np.mean(np.array(particles[:args.n_particles]), axis=0)
    centers = [centers[args.axes[0]], centers[args.axes[1]], centers[args.axes[2]]]
    if focus:
        r = args.plot_radius# (int(np.sqrt(args.floor_dim)) - 1) * args.plot_scale / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    return box_patches

def anim_box(args, env_config, obs_seq, pred_transformations, save_path='', data_path=''):
    title_list = ['Ground truth', 'Prediction']
    boxes = []
    for obj_name, n_particles_per_object in args.n_particles_per_object.items():
        # Initial box properties (positions, quaternions, sizes)
        if 'cube' in obj_name:
            if 'cubeB' in obj_name and int(obj_name[-1]) >= env_config['num_obj_on_shelf']:
                continue
            box_position = obs_seq[0][f'{obj_name}_pos']
            box_quat = obs_seq[0][f'{obj_name}_quat']
            box_size = env_config[f'{obj_name}_size']
        elif 'gripper' in obj_name:
            continue
        else:
            box_position = env_config[f'{obj_name}']['pos']
            box_quat = env_config[f'{obj_name}']['quat']
            box_size = env_config[f'{obj_name}']['size']
        
        box = pv.Cube([0, 0, 0], x_length=box_size[0], y_length=box_size[1], z_length=box_size[2])
        homo_transform = np.concatenate((np.concatenate(
                (trans.quat2mat(box_quat), np.array(box_position).reshape(3, 1)), axis=1), np.array([0., 0., 0., 1.]).reshape(1, 4)), axis=0)
        box.transform(homo_transform)
        boxes.append(box)
        
    # Create the plotter
    plotter = BackgroundPlotter()

    # Add initial boxes to plotter
    for box in boxes:
        plotter.add_mesh(box)
    num_dynamic_objects = env_config['num_obj_on_shelf']+1

    # Create a function to update the plot at each frame
    def update(frame):
        plotter.clear()  # Clear previous boxes
        for i, box in enumerate(boxes[:num_dynamic_objects]):
            # Update position and rotation
            homo = np.concatenate((np.concatenate(
                (pred_transformations[frame]['rotation'][i].cpu().numpy(), pred_transformations[frame]['translation'][i].cpu().numpy().reshape(3, 1)), axis=1), np.array([0., 0., 0., 1.]).reshape(1, 4)), axis=0)
            # box.translate(pred_transformations[i][frame]['translation'])
            box.transform(homo)
            # Add updated box to plotter
            plotter.add_mesh(box)
    # Create the animation
    plotter.open_gif(f"{save_path}.gif", fps=5)
    for i in range(len(pred_transformations)):
        plotter.update()
        update(i)
        plotter.write_frame()
    plotter.close()

    # Convert gif to mp4
    import moviepy.editor as mp
    clip = mp.VideoFileClip(f"{save_path}.gif")
    clip.write_videofile(f"{save_path}.mp4")

def render_anim(args, col_titles, state_seqs, attn_mask_pred=None, rels_pred=None, draw_set=['dough', 'floor', 'tool'],
                axis_off=False, target=None, views=[(90, -90), (0, -90), (45, -45)], fps=1, img_st_idx=0, res='high', path='', only_goal_img=False,
                data_path=''):
    
    if data_path != '' :
        env_config = load_json(os.path.join(data_path, "env_config.json"))
    else:
        env_config = None

    cam_name = 'cam_1' if not only_goal_img else 'agentview'
    n_frames = max([x.shape[0] for x in state_seqs])
    n_rows = 1
    if args.show_src_img:
        n_cols = len(col_titles) + 1
        col_titles.insert(0, 'Camera')
    else:
        n_cols = len(col_titles) 

    fig_size = 5 if res == 'high' else 3
    title_fontsize = 30 if res == 'high' else 10
    # Set the maximum width of each title line
    max_title_width = 20

    fig, big_axes = plt.subplots(1, 1, figsize=(fig_size * n_cols, fig_size * n_rows))
    sm = cm.ScalarMappable(cmap='plasma')

    plot_info_dict = {}
    for source_idx in range(n_cols):
        target_cur = target[source_idx] if isinstance(target, list) else target

        ax_cur = big_axes

        # ax_cur.set_title(col_titles[i], fontweight='semibold', fontsize=title_fontsize)
        ax_cur.axis('off')

        plot_info = []
        if args.show_src_img:
            if source_idx == 0:
                if only_goal_img:
                    image_paths = [os.path.join(data_path, f'{str(img_st_idx+1).zfill(3)}_rgb_{cam_name}.png')]
                else:
                    # Get the list of image files in the directory
                    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(f'_rgb_{cam_name}.png')])
                    # Determine the actual number of frames available
                    n_frames_available = len(image_files) - img_st_idx
                    # Set n_frames to the minimum of either the requested number of frames or the actual number of frames available
                    n_img_frames = min(n_frames, n_frames_available)
                    # Construct a list of image paths for the requested frames
                    image_paths = [os.path.join(data_path, f'{str(img_idx).zfill(3)}_rgb_{cam_name}.png') for img_idx in range(img_st_idx, img_st_idx+n_img_frames)]
                images = [plt.imread(image_path) for image_path in image_paths]
                ax = fig.add_subplot(n_rows, n_cols, source_idx+1)
                image_plot = ax.imshow(images[0])
                if axis_off:
                    ax.axis('off')
                
                plot_info.append((ax, image_plot))
            else:
                ax = fig.add_subplot(n_rows, n_cols, source_idx+1, projection='3d')

                ax.view_init(*views[-1])
                
                if source_idx == n_cols-1 and rels_pred is not None:
                    rels = rels_pred[0]
                else:
                    rels = None
                if args.show_box:
                    outputs = visualize_boxes(ax, args, state_seqs[source_idx-1][0], target_cur,
                                            axis_off=axis_off, mask=None, rels=rels, res=res, env_config=env_config)
                else:
                    outputs = visualize_points(ax, args, state_seqs[source_idx-1][0], target_cur,
                                            axis_off=axis_off, mask=None, rels=rels, res=res, env_config=env_config)

                plot_info.append((ax, outputs))
            wrapped_title = textwrap.wrap(col_titles[source_idx], width=max_title_width)

            ax.set_title('\n'.join(wrapped_title), fontweight='semibold', fontsize=title_fontsize)

            plot_info_dict[col_titles[source_idx]] = plot_info

    def update(step):
        outputs_all = []
        for source_idx in range(n_cols):
            if source_idx == 0:
                step_cur = min(step, len(images) - 1)
            else:
                state = state_seqs[source_idx-1]
                step_cur = min(step, state.shape[0] - 1)
                frame_cur = state[step_cur]
            if rels_pred is None:
                rels = None
            else:
                rels = rels_pred[step_cur]
            for j in range(n_rows):
                ax, outputs = plot_info_dict[col_titles[source_idx]][j]

                if args.show_src_img:
                    if source_idx == 0:
                        if step_cur < len(images):
                            image_plot.set_data(images[step_cur])
                        else:
                            image_plot.set_data(images[-1])

                    #     if step_cur < len(images):
                    #         image_plot = images[step_cur]
                    #     else:
                    #         image_plot = images[-1]
                    # if image_plot is not None:  # If image_plot has been set, update the image
                    #     image_plot.set_data(images[step_cur])
                    else:
                        if args.show_box:
                            # Remove old boxes from the axis
                            for box_patch in outputs:
                                box_patch.remove()

                            # Call visualize_boxes to create new boxes for the current frame
                            box_patches = visualize_boxes(ax, args, state_seqs[source_idx-1][step_cur], target_cur,
                                                        axis_off=axis_off, mask=None, rels=rels, res=res, env_config=env_config)
                            
                            # Update the plot_info_dict with the new box_patches
                            plot_info_dict[col_titles[source_idx]][j] = (ax, box_patches)
                        else:
                            particles_idx = 0
                            for draw_idx, (obj_name, n_particles_per_object) in enumerate(args.n_particles_per_object_clean.items()):

                                outputs[draw_idx]._offsets3d = (
                                    frame_cur[particles_idx:particles_idx + n_particles_per_object, args.axes[0]],
                                    frame_cur[particles_idx:particles_idx + n_particles_per_object, args.axes[1]],
                                    frame_cur[particles_idx:particles_idx + n_particles_per_object, args.axes[2]]
                                )
                                particles_idx += n_particles_per_object
                            outputs_all.extend(outputs)

                else:
                    particles_idx = 0
                    for draw_idx, (obj_name, n_particles_per_object) in enumerate(args.n_particles_per_object_clean.items()):

                        outputs[draw_idx]._offsets3d = (
                            frame_cur[particles_idx:particles_idx + n_particles_per_object, args.axes[0]],
                            frame_cur[particles_idx:particles_idx + n_particles_per_object, args.axes[1]],
                            frame_cur[particles_idx:particles_idx + n_particles_per_object, args.axes[2]]
                        )
                        particles_idx += n_particles_per_object

                    outputs_all.extend(outputs)
                if source_idx == n_cols - 1 and rels_pred is not None:
                    num_rels = rels.shape[0]
                    for k in range(num_rels):
                        if rels[k][0] > 0 and rels[k][1] > 0:
                            outputs[k-num_rels].set_data_3d(frame_cur[rels[k], args.axes[0]], frame_cur[rels[k], args.axes[1]], 
                                frame_cur[rels[k], args.axes[2]])
                        else:
                            outputs[k-num_rels].set_data_3d([], [], [])
                    outputs_all.extend(outputs)

        return outputs_all

    anim = animation.FuncAnimation(
        fig, update, frames=np.arange(0, n_frames), interval=100, blit=True)

    if len(path) > 0:
        anim.save(path, writer=animation.FFMpegWriter(fps=fps))
        
        # Save the last frame as an image
        last_frame = anim.save_count - 1  # Get the index of the last frame
        anim._draw_frame(last_frame)  # Draw the last frame on the figure
        image_path = path.replace('anim', 'image').replace('mp4', 'png')
        plt.savefig(image_path)  # Save the last frame as an image
    else:
        plt.show()

    plt.close()


def render_frames(args, row_titles, state_seq, frame_list=[], axis_off=True, focus=True,
                  draw_set=['dough', 'floor', 'tool'], target=None, views=[(90, -90), (0, -90), (45, -45)],
                  res='high', path='', name=''):

    n_frames = state_seq[1].shape[0]
    n_rows = len(row_titles)
    n_cols = len(views)

    fig_size = 12 if res == 'high' else 3
    title_fontsize = 60 if res == 'high' else 10
    fig, big_axes = plt.subplots(n_rows, 1, figsize=(
        fig_size * n_cols, fig_size * n_rows))

    if len(frame_list) == 0:
        frame_list = range(n_frames)

    for frame in frame_list:
        for i in range(n_rows):
            state = state_seq[i]
            target_cur = target[i] if isinstance(target, list) else target
            focus_cur = focus[i] if isinstance(focus, list) else focus
            if n_rows == 1:
                big_axes.set_title(
                    row_titles[i], fontweight='semibold', fontsize=title_fontsize)
                big_axes.axis('off')
            else:
                big_axes[i].set_title(
                    row_titles[i], fontweight='semibold', fontsize=title_fontsize)
                big_axes[i].axis('off')

            for j in range(n_cols):
                ax = fig.add_subplot(n_rows, n_cols, i *
                                     n_cols + j + 1, projection='3d')
                ax.view_init(*views[j])
                visualize_points(ax, args, state[0] if state.shape[0] == 1 else state[frame], target_cur, axis_off=axis_off, focus=focus_cur, res=res)

        # plt.tight_layout()

        if len(path) > 0:
            if len(name) == 0:
                plt.savefig(os.path.join(path, f'{str(frame).zfill(3)}.pdf'))
            else:
                plt.savefig(os.path.join(path, name))
        else:
            plt.show()

    plt.close()


def render_processed_pcd(args, points, obs, axis_off=True, focus=True, res='low', path='', camera_names=None):
    import matplotlib.gridspec as gridspec
    if camera_names is None:
        pass
    views = [(30, -30), (30, 30)]
    n_rows = len(camera_names)
    n_cols = 3
    col_titles = ['Camera', 'Processed Point Cloud']
    
    fig_size = 12 if res == 'high' else 5
    title_fontsize = 60 if res == 'high' else 10
    # Create a GridSpec with 2 rows and 2 columns
    width_ratios = [1.6] + [n_rows] * (n_cols - 1)
    height_ratios = [1] * n_rows
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.2,  # Increase space between columns
        hspace=0.2,  # Increase space between rows
    )

    # Create the figure
    fig = plt.figure(figsize=(fig_size * n_cols, fig_size * n_rows))

    # Add column titles
    # fig.suptitle(col_titles[0], x=0.25, y=0.95, fontsize=title_fontsize, ha='center')
    # fig.suptitle(col_titles[1], x=0.75, y=0.95, fontsize=title_fontsize, ha='center')

    for j in range(n_cols):

        if j == 0:
            for i in range(n_rows):
                # Create the subplots
                ax = plt.subplot(gs[i, 0])  # i-th row, first column
                ax.imshow(obs[f"{camera_names[i]}_image"][..., ::-1])
                ax.axis('off')
        else:
            ax = plt.subplot(gs[:, j], projection='3d')  # Both rows, second column
            ax.view_init(*(views[j-1]))
            visualize_points(ax, args, points, target=None, axis_off=False, focus=focus, res=res)

    plt.tight_layout()

    if len(path) > 0:
        plt.savefig(path, dpi=300, pad_inches=0.1)
    else:
        plt.show()

    plt.close()


def render_o3d(geometry_list, axis_off=False, focus=True, views=[(90, -90), (0, -90), (45, -45)], label_list=[], point_size_list=[], path=''):
    n_rows = 2
    n_cols = 3

    fig, big_axes = plt.subplots(n_rows, 1, figsize=(12 * n_cols, 12 * n_rows))
    sm = cm.ScalarMappable(cmap='plasma')

    for i in range(n_rows):
        ax_cur = big_axes[i]

        title_fontsize = 60
        ax_cur.set_title('Test', fontweight='semibold',
                         fontsize=title_fontsize)
        ax_cur.axis('off')

        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i *
                                 n_cols + j + 1, projection='3d')
            ax.computed_zorder = False
            ax.view_init(*views[j])
            if j == n_cols - 1:
                fig.colorbar(mappable=sm, ax=ax)

            for k in range(len(geometry_list)):
                type = geometry_list[k].get_geometry_type()
                # Point Cloud
                # if type == o3d.geometry.Geometry.Type.PointCloud:
                #     geometry.paint_uniform_color(pcd_color)
                # Triangle Mesh
                if type == o3d.geometry.Geometry.Type.TriangleMesh:
                    mf = pymeshfix.MeshFix(np.asarray(
                        geometry_list[k].vertices), np.asarray(geometry_list[k].triangles))
                    mf.repair()
                    mesh = mf.mesh
                    vertices = np.asarray(mesh.points)
                    triangles = np.asarray(mesh.faces).reshape(
                        mesh.n_faces, -1)[:, 1:]
                    ax.plot_trisurf(
                        vertices[:, 0], vertices[:, 1], triangles=triangles, Z=vertices[:, 2])
                    # ax.set_aspect('equal')
                elif type == o3d.geometry.Geometry.Type.PointCloud:
                    particles = np.asarray(geometry_list[k].points)
                    colors = np.asarray(geometry_list[k].colors)
                    if len(point_size_list) > 0:
                        point_size = point_size_list[k]
                    else:
                        point_size = 160
                    if len(label_list) > 0:
                        label = label_list[k]
                        if 'dough' in label:
                            ax.scatter(
                                particles[:, 0], particles[:, 1], particles[:, 2], c='b', s=point_size, label=label)
                        elif 'tool' in label:
                            ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                                       c='r', alpha=0.2, zorder=4.2, s=point_size, label=label)
                        else:
                            ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                                       c='yellowgreen', zorder=4.1, s=point_size, label=label)
                    else:
                        label = None
                        ax.scatter(
                            particles[:, 0], particles[:, 1], particles[:, 2], c=colors, s=point_size, label=label)
                else:
                    raise NotImplementedError

            if axis_off:
                ax.axis('off')

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            if len(label_list) > 0:
                ax.legend(fontsize=30, loc='upper right',
                          bbox_to_anchor=(0.0, 0.0))

            # extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            # size = extents[:, 1] - extents[:, 0]
            centers = geometry_list[0].get_center()
            if focus:
                r = 0.05
                for ctr, dim in zip(centers, 'xyz'):
                    getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    # plt.tight_layout()

    if len(path) > 0:
        plt.savefig(f'{path}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')
    else:
        plt.show()

    plt.close()


def visualize_o3d(args, geometry_list, title='O3D', view_point=None, point_size=5, pcd_color=[0, 0, 0],
                  mesh_color=[0.5, 0.5, 0.5], show_normal=False, show_frame=True, path=''):
    '''
    show o3d.geometry.Geometry
    args:
    geometry_list: list of o3d.geometry.Geometry
    title: str
    view_point: list of float
    point_size: float
    pcd_color: list of float
    mesh_color: list of float
    show_normal: bool
    show_frame: bool
    path: str
    '''
    if args.evenly_spaced:
        # Create a canvas and a 3D view
        canvas = SceneCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        # Create a markers visual to display the points
        points = np.concatenate(geometry_list, axis=0)
        
        # Generate a colormap and assign unique colors to each element in geometry_list
        colormap = cm.get_cmap('viridis', len(geometry_list))
        face_colors = []
        for i, geometry in enumerate(geometry_list):
            face_colors.extend([colormap(i)[:3] + (0.5,) for _ in geometry])
        face_colors = np.array(face_colors)

        markers = Markers()
        markers.set_data(points, edge_color=None,
                         face_color=(1, 1, 1, 0.5), size=5)

        # Add the markers visual to the view
        view.add(markers)

        # Run the event loop
        app.run()

    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(title)
        types = []

        for geometry in geometry_list:
            type = geometry.get_geometry_type()
            # Point Cloud
            # if type == o3d.geometry.Geometry.Type.PointCloud:
            #     geometry.paint_uniform_color(pcd_color)
            # Triangle Mesh
            if type == o3d.geometry.Geometry.Type.TriangleMesh:
                geometry.paint_uniform_color(mesh_color)
            types.append(type)

            vis.add_geometry(geometry)
            vis.update_geometry(geometry)

        vis.get_render_option().background_color = np.array([0., 0., 0.])
        if show_frame:
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
            vis.add_geometry(mesh)
            vis.update_geometry(mesh)

        if o3d.geometry.Geometry.Type.PointCloud in types:
            vis.get_render_option().point_size = point_size
            vis.get_render_option().point_show_normal = show_normal
        if o3d.geometry.Geometry.Type.TriangleMesh in types:
            vis.get_render_option().mesh_show_back_face = True
            vis.get_render_option().mesh_show_wireframe = True

        vis.poll_events()
        vis.update_renderer()

        # if view_point is None:
        #     vis.get_view_control().set_front(np.array([0.305, -0.463, 0.832]))
        #     vis.get_view_control().set_lookat(np.array([0.4, -0.1, 0.0]))
        #     vis.get_view_control().set_up(np.array([-0.560, 0.620, 1.550]))
        #     vis.get_view_control().set_zoom(0.3)
        # else:
        #     vis.get_view_control().set_front(view_point['front'])
        #     vis.get_view_control().set_lookat(view_point['lookat'])
        #     vis.get_view_control().set_up(view_point['up'])
        #     vis.get_view_control().set_zoom(view_point['zoom'])

        # cd = os.path.dirname(os.path.realpath(sys.argv[0]))
        # path = os.path.join(cd, '..', 'figures', 'images', f'{title}_{datetime.now().strftime("%b-%d-%H:%M:%S")}.png')

        if len(path) > 0:
            vis.capture_screen_image(path, True)
            vis.destroy_window()
        else:
            vis.run()


def visualize_target(args, target_shape_name):
    target_frame_path = os.path.join(os.getcwd(), 'target_shapes',
                                     target_shape_name, f'{target_shape_name.split("/")[-1]}.h5')
    visualize_h5(args, target_frame_path)


def visualize_h5(args, file_path):
    hf = h5py.File(file_path, 'r')
    data = []
    for i in range(len(args.data_names)):
        d = np.array(hf.get(args.data_names[i]))
        data.append(d)
    hf.close()
    target_shape = data[0][:args.n_particles, :]
    render_frames(args, ['H5'], [np.array([target_shape])], draw_set=['dough'])


def visualize_neighbors(args, particles, target, neighbors, path=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # red is the target and blue are the neighbors
    ax.scatter(particles[:args.n_particles, args.axes[0]], particles[:args.n_particles, args.axes[1]],
               particles[:args.n_particles, args.axes[2]], c='c', alpha=0.2, s=30)
    ax.scatter(particles[args.n_particles:, args.axes[0]], particles[args.n_particles:, args.axes[1]],
               particles[args.n_particles:, args.axes[2]], c='r', alpha=0.2, s=30)

    ax.scatter(particles[neighbors, args.axes[0]], particles[neighbors,
               args.axes[1]], particles[neighbors, args.axes[2]], c='b', s=60)
    ax.scatter(particles[target, args.axes[0]], particles[target,
               args.axes[1]], particles[target, args.axes[2]], c='r', s=60)

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))()
                       for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def plot_cm(test_set, y_true, y_pred, path=''):
    confusion_matrix = metrics.confusion_matrix([test_set.classes[x] for x in y_true],
                                                [test_set.classes[x] for x in y_pred], labels=test_set.classes)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=test_set.classes)
    cm_display.plot(xticks_rotation='vertical')
    plt.gcf().set_size_inches(12, 12)
    plt.tight_layout()
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def concat_images(imga, imgb, direction='h'):
    # combines two color image ndarrays side-by-side.
    ha, wa = imga.shape[:2]
    hb, wb = imgb.shape[:2]

    if direction == 'h':
        max_height = np.max([ha, hb])
        total_width = wa + wb
        new_img = np.zeros(
            shape=(max_height, total_width, 3), dtype=imga.dtype)
        new_img[:ha, :wa] = imga
        new_img[:hb, wa:wa+wb] = imgb
    else:
        max_width = np.max([wa, wb])
        total_height = ha + hb
        new_img = np.zeros(
            shape=(total_height, max_width, 3), dtype=imga.dtype)
        new_img[:ha, :wa] = imga
        new_img[ha:ha+hb, :wb] = imgb

    return new_img


def concat_n_images(image_path_list, n_rows, n_cols):
    # combines N color images from a list of image paths
    row_images = []
    for i in range(n_rows):
        output = None
        for j in range(n_cols):
            idx = i * n_cols + j
            img_path = image_path_list[idx]
            img = plt.imread(img_path)[:, :, :3]
            if j == 0:
                output = img
            else:
                output = concat_images(output, img)
        row_images.append(output)

    output = row_images[0]
    # row_images.append(abs(row_images[1] - row_images[0]))
    for img in row_images[1:]:
        output = concat_images(output, img, direction='v')

    return output


def visualize_image_pred(img_paths, target, output, classes, path=''):
    concat_imgs = concat_n_images(img_paths, n_rows=2, n_cols=4)
    plt.imshow(concat_imgs)

    pred_str = ', '.join([classes[x] for x in output])
    plt.text(10, -30, f'prediction: {pred_str}', c='black')
    if target is not None:
        plt.text(10, -60, f'label: {classes[target]}', c='black')
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_pcd_pred(row_titles, state_list, views=[(90, -90), (0, -90), (45, -45)], axis_off=False, res='low', path=''):
    n_rows = len(row_titles)
    n_cols = len(views)

    fig_size = 12 if res == 'high' else 3
    title_fontsize = 60 if res == 'high' else 10
    point_size = 160 if res == 'high' else 10
    fig, big_axes = plt.subplots(n_rows, 1, figsize=(
        fig_size * n_cols, fig_size * n_rows))

    for i in range(n_rows):
        state = state_list[i]
        if n_rows == 1:
            big_axes.set_title(
                row_titles[i], fontweight='semibold', fontsize=title_fontsize)
            big_axes.axis('off')
        else:
            big_axes[i].set_title(
                row_titles[i], fontweight='semibold', fontsize=title_fontsize)
            big_axes[i].axis('off')

        for j in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i *
                                 n_cols + j + 1, projection='3d')
            ax.view_init(*views[j])
            state_colors = state[:, 3:6] if state.shape[1] > 3 else 'b'
            ax.scatter(state[:, 0], state[:, 1], state[:, 2],
                       c=state_colors, s=point_size)
            # ax.set_zlim(-0.075, 0.075)

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            if axis_off:
                ax.axis('off')

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def visualize_tensor(tensor, path='', mode='RGB'):
    im = np.array(transforms.ToPILImage()(tensor[:3]))
    # if mode == 'HSV':
    #     im_array = np.array(im)
    #     im = cv.cvtColor(im_array, cv.COLOR_HSV2RGB)

    for i in range(3, tensor.shape[0], 3):
        # import pdb; pdb.set_trace()
        im_next = np.array(transforms.ToPILImage()(tensor[i:i+3]))
        # if mode == 'HSV':
        #     im_next_array = np.array(im_next)
        #     im_next = cv.cvtColor(im_next_array, cv.COLOR_HSV2RGB)
        im = concat_images(im, im_next)
    plt.imshow(im)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()


def main():
    if len(sys.argv) < 2:
        print('Please specify the path of the pickle file!')
        exit()

    pkl_path = sys.argv[1]
    with open(pkl_path, 'rb') as f:
        args_dict = pickle.load(f)
        anim_path = pkl_path.replace('_args', '').replace('pkl', 'mp4')
        print(f'Rendering anim at {anim_path}...')
        render_anim(**args_dict, path=anim_path)
        print('Animation rendered!')


if __name__ == '__main__':
    main()
