import numpy as np
import robosuite as suite
from robosuite import load_controller_config
import matplotlib.pyplot as plt

# controllers = {
#     "OSC_POSITION": [4, 3, 0.1],
# }

# Define controller path to load
controller_config = load_controller_config(default_controller="OSC_POSITION")

# create environment instance
env = suite.make(
    env_name="StowInsert", # try with other tasks like "Stack" and "Door"
    robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    # render_camera="cam_1",
    has_offscreen_renderer=False,
    use_camera_obs=False,
    controller_configs=controller_config,
    camera_names="cam_1",
    camera_heights=256,
    camera_widths=256,
    camera_depths=False,
    camera_segmentations=None,  # {None, instance, class, element}
    renderer="mujoco",
)

# reset the environment
env.reset()

for i in range(40):
    while True:
        # action = np.zeros(4) # sample random action
        action = np.array([-0., 0.1, 0.9, 0])
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
        if done:
            break

env.close()