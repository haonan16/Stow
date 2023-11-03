"""
This file implements a wrapper for saving simulation states to disk.
This data collection wrapper is useful for collecting demonstrations.
"""

import os
import time
import numpy as np

from robosuite.wrappers import Wrapper
from robosuite.utils.mjcf_utils import save_sim_model
from utils.data_utils import *
import pdb
from PIL import Image
from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
import glfw


class KeyframeCollectionWrapper(Wrapper):
    def __init__(self, env, directory, collect_freq=1, flush_freq=100):
        """
        Initializes the data collection wrapper.

        Args:
            env (MujocoEnv): The environment to monitor.
            directory (str): Where to store collected data.
            collect_freq (int): How often to save simulation state, in terms of environment steps.
            flush_freq (int): How frequently to dump data to disk, in terms of environment steps.
        """
        super().__init__(env)

        # the base directory for all logging
        self.directory = directory

        # in-memory cache for simulation states and action info
        self.states = []
        self.action_infos = []  # stores information about actions taken
        self.obs = []

        # how often to save simulation state, in terms of environment steps
        self.collect_freq = collect_freq

        # how frequently to dump data to disk, in terms of environment steps
        self.flush_freq = flush_freq

        if not os.path.exists(directory):
            print("DataCollectionWrapper: making new directory at {}".format(directory))
            os.makedirs(directory)

        # store logging directory for current episode
        self.ep_directory = None

        # remember whether any environment interaction has occurred
        self.has_interaction = False

        # some variables for remembering the current episode's initial state and model xml
        self._current_task_instance_state = None
        self._current_task_instance_xml = None
        
        self.frame_idx = 0
        self.ep_idx = -1

    def _start_new_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """
        

        # flush any data left over from the previous episode if any interactions have happened
        if self.has_interaction:
            self._flush()

        # timesteps in current episode
        self.t = 0
        self.has_interaction = False

        # save the task instance (will be saved on the first env interaction)
        self._current_task_instance_xml = self.env.sim.model.get_xml()
        self._current_task_instance_state = np.array(self.env.sim.get_state().flatten())
        
        
        # trick for ensuring that we can play MuJoCo demonstrations back
        # deterministically by using the recorded actions open loop
        self.env.reset_from_xml_string(self._current_task_instance_xml)
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(self._current_task_instance_state)
        self.env.sim.forward()

        
    def _on_first_interaction(self):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).

        Raises:
            AssertionError: [Episode path already exists]
        """

        self.has_interaction = True

        # create a directory with a timestamp
        t1, t2 = str(time.time()).split(".")
        self.ep_directory = os.path.join(self.directory, f"ep_{self.ep_idx:03d}")
        assert not os.path.exists(self.ep_directory)
        print("DataCollectionWrapper: making folder at {}".format(self.ep_directory))
        os.makedirs(self.ep_directory)

        # save the model xml
        xml_path = os.path.join(self.ep_directory, "model.xml")
        with open(xml_path, "w") as f:
            f.write(self._current_task_instance_xml)

        store_json(self.env_config, os.path.join(self.ep_directory, "env_config.json"))

        # save initial state and action
        assert len(self.states) == 0
        self.states.append(self._current_task_instance_state)
        

    def _flush(self):
        """
        Method to flush internal state to disk.
        """
        t1, t2 = str(time.time()).split(".")
        state_path = os.path.join(self.ep_directory, f"{self.frame_idx:03d}_state.npz")
        if hasattr(self.env, "unwrapped"):
            env_name = self.env.unwrapped.__class__.__name__
        else:
            env_name = self.env.__class__.__name__
        np.savez(
            state_path,
            states=np.array(self.states),
            obs=self.obs,
            action_infos=self.action_infos,
            env=env_name,
        )
        self.states = []
        self.action_infos = []
        self.obs = []

        # Get the current OpenGL viewport size
        width, height = glfw.get_framebuffer_size(self.sim.render_contexts[0].window)

        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

        # Convert the pixel data to a NumPy array
        img_array = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)[::-1]

        # Save the image using PIL
        img = Image.fromarray(img_array)
        img.save(os.path.join(self.ep_directory, f"{self.frame_idx:03d}_rgb_agentview.png"))
        self.frame_idx += 1

        print('KeyframeCollectionWrapper: data flushed to disk')

    def reset(self):
        """
        Extends vanilla reset() function call to accommodate data collection

        Returns:
            OrderedDict: Environment observation space after reset occurs
        """
        ret = super().reset()
        self._start_new_episode()
        
        self._initial_obs = ret
        # save initial obs 
        assert len(self.obs) == 0
        self.obs.append(self._initial_obs)
        
        self.ep_idx += 1
        self.frame_idx = 0

        return ret

    def step(self, action, user_input=None):
        """
        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """

        ret = super().step(action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        if user_input == 's':
            print(f"User input: {user_input} received!")

            # collect the current simulation state if necessary

            state = self.env.sim.get_state().flatten()
            self.states.append(state)
            self.obs.append(ret[0])

            info = {}
            info["actions"] = np.array(action)
            self.action_infos.append(info)

            # flush collected data to disk if necessary
            self._flush()
            
            # print("KeyframeCollectionWrapper: state saved")

        return ret[0]

    def close(self):
        """
        Override close method in order to flush left over data
        """
        if self.has_interaction:
            self._flush()
        self.env.close()
