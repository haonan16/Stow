#!/usr/bin/env python3

import os
import glob
import shutil
import subprocess
from skills.skill_controller import SkillController

# Pass skill_name and perception_dir as arguments
skill_name = "insert"
perception_dir = "./dump/perception/Stow_{}/23-May-2023-12:35:57.698997_corner".format(skill_name)
inspect_txt = "./dump/perception/inspect/{}/inspect.txt".format(skill_name)

# Create an "archive" subdirectory
archive_dir = os.path.join(perception_dir, "archive")
os.makedirs(archive_dir, exist_ok=True)

# Archive the directories specified in the inspect.txt file
with open(inspect_txt, 'r') as f:
    for line in f:
        idx = line.strip()
        print("Archiving: {}".format(idx))
        shutil.move(os.path.join(perception_dir, idx), os.path.join(archive_dir, idx))

# Move subdirectories with 'invalid_sample.txt' to the "archive" directory
sub_dirs = glob.glob(os.path.join(perception_dir, "*[0-9]*"))
for sub_dir in sub_dirs:
    invalid_file = os.path.join(sub_dir, "invalid_sample.txt")
    num_h5_files = len(glob.glob(os.path.join(sub_dir, "*.h5")))
    if os.path.isfile(invalid_file) or num_h5_files != SkillController.SKILL_MAPS[skill_name].num_keyframes():
        print("Archiving: {}".format(sub_dir))
        shutil.move(sub_dir, archive_dir)


# Find the last directory index
sub_dirs = glob.glob(os.path.join(perception_dir, "*[0-9]*"))
last_dir = max(int(os.path.basename(sub_dir)) for sub_dir in sub_dirs)

# Patch the directory structure by moving directories with higher indices
for i in range(last_dir + 1):
    sub_dir = os.path.join(perception_dir, "{:03d}".format(i))
    if not os.path.isdir(sub_dir):
        # Find the next available directory with a higher index
        for j in range(i + 1, last_dir + 1):
            next_sub_dir = os.path.join(perception_dir, "{:03d}".format(j))
            if os.path.isdir(next_sub_dir):
                print("Patching: {} -> {}".format(next_sub_dir, sub_dir))
                shutil.move(next_sub_dir, sub_dir)
                break

# Call main() function with specified arguments
subprocess.call(["python", "perception/scripts/make_dataset.py", skill_name, perception_dir])
