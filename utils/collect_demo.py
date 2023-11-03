"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np
import json

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper, KeyframeCollectionWrapper
from robosuite.utils.input_utils import input2action
from config.config import gen_args

import threading

import pdb
from PIL import Image
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
import glfw

user_input = None

def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """
    global user_input

    env.reset()
    

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    while True:
        # Set active robot
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break
        # zero_action = np.array([ 0.,  0., -0.,  0.,  0., -0., -1.])
        # if np.all(zero_action==action) and not is_first:
        #     pass
        # else:
        #     # Run environment step
        #     env.step(action)
        #     is_first = False
        # else:
        #     print("Main loop is running...")
        # env.step(action, user_input)

        if user_input == "s":
            env.step(action, user_input)
            user_input = None
        else:
            env.step(action)
            pass
        
        env.render()



    # cleanup for end of data collection episodes
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file. 
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
                
        if len(states) == 0:
            continue

        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    f.close()

def process_keyboard_input():
    global user_input

    while True:
        user_input = input("Enter a message (s for save):\n ")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Stow")
    parser.add_argument("--robots", nargs="+", type=str, default="Kinova3", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'")
    parser.add_argument("--device", type=str, default="spacemouse", help="Which device to use for collecting demos")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()


    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)
    controller_config.update({'control_delta': True})

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    args_env = gen_args()
    # Create environment
    env = suite.make(
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=args.camera,
        args=args_env,
        ignore_done=True,
        control_freq=20,
        **config,
        )


    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    time_stamp = str(time.time()).replace(".", "_")
    # tmp_directory = "{}/{}".format(os.path.join(suite.models.assets_root, "demonstrations"), time_stamp, "raw")
    demo_directory = os.path.join(args_env.demo_path, "{}".format(time_stamp))  
    # new_dir = os.path.join("tmp", time_stamp)
    h5_dir = os.path.join(demo_directory)

    env = KeyframeCollectionWrapper(env, demo_directory)

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    # t1, t2 = str(time.time()).split(".")
    # os.makedirs(new_dir)
    
    # Create a separate thread for keyboard input processing
    input_thread = threading.Thread(target=process_keyboard_input, daemon=True)
    input_thread.start()

    # collect demonstrations
    while True:
        collect_human_trajectory(env, device, args.arm, args.config)
        # gather_demonstrations_as_hdf5(demo_directory, h5_dir, env_info)
        