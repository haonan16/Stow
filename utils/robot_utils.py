import numpy as np


def is_gripper_closed(gripper_joint_values, threshold=0.1):
    left_knuckle_joint_value = gripper_joint_values[0]
    right_knuckle_joint_value = gripper_joint_values[3]
    
    # Check if the gripper is open by comparing the joint values to a threshold
    if np.abs(left_knuckle_joint_value) > threshold and np.abs(right_knuckle_joint_value) > threshold:
        return True
    else:
        return False

def get_gripper_width(gripper_joint_values, threshold=0.1, closed_width=0.05, open_width=0.085):
    """
    Determine the gripper width based on the joint values.
    
    Args:
        gripper_joint_values (np.array): Gripper joint values (left and right knuckle)
        threshold (float, optional): Joint value threshold to consider the gripper closed. Defaults to 0.1.
        closed_width (float, optional): Gripper width when closed. Defaults to 0.03.
        open_width (float, optional): Gripper width when open. Defaults to 0.085.

    Returns:
        float: Gripper width (either closed_width or open_width)
    """
    left_knuckle_joint_value = gripper_joint_values[0]
    right_knuckle_joint_value = gripper_joint_values[3]

    if np.abs(left_knuckle_joint_value) > threshold and np.abs(right_knuckle_joint_value) > threshold:
        return closed_width
    else:
        return open_width
