import glob
import numpy as np
import os
import sys
import argparse

def main(skill_name, perception_dir):
    # skill_name = 'push'
    # perception_dir=[
    #     "./dump/perception/Stow_"+"{}/raw/12-Apr-2023-10:42:46.189469".format(skill_name),
    # ]
    # push, 28-Mar-2023-16:22:35.712825
    # sweep, 11-Apr-2023-09:00:51.536277

    # tool_type += '_surf_nocorr'
    cd = os.path.dirname(os.path.realpath(sys.argv[0]))
    data_root_dir = os.path.join(cd, "..", "..", "data", f"gt/stow_{skill_name}")
    if type(perception_dir) is list:
        perception_data_list = []
        for p_dir in perception_dir:
            p_dir = os.path.join(cd, "..", "..", p_dir)
            perception_data_list += sorted(glob.glob(os.path.join(p_dir, '*')))
    else:
        perception_dir = os.path.join(cd, "..", "..", perception_dir)
        perception_data_list = sorted(glob.glob(os.path.join(perception_dir, '*')))

    np.random.seed(0)
    np.random.shuffle(perception_data_list)
    for path in perception_data_list:
        if 'archive' in path: 
            perception_data_list.remove(path)

    dataset_size = len(perception_data_list)
    valid_set_size = int(dataset_size * 0.1)
    test_set_size = int(dataset_size * 0.1)
    training_set_size = dataset_size - valid_set_size - test_set_size

    print(f"Training set size: {training_set_size}")
    print(f"Valid set size: {valid_set_size}")
    print(f"Test set size: {test_set_size}")
    
    dataset_dict = {"train": training_set_size, "valid": valid_set_size, "test": test_set_size}
    p_idx = 0
    
    for dataset, size in dataset_dict.items():
        dataset_dir = os.path.join(data_root_dir, dataset)
        if not os.path.exists(dataset_dir):
            os.system('mkdir -p ' + dataset_dir)

        existing_data = sorted(glob.glob(os.path.join(dataset_dir, '*')))
        starting_idx = len(existing_data)

        for i in range(starting_idx, starting_idx + size):
            p_name = os.path.basename(perception_data_list[p_idx])
            data_name = str(i).zfill(3)
            print(f'{p_name} -> {data_name}')
            os.system(f'cp -r {perception_data_list[p_idx]} {os.path.join(dataset_dir, data_name)}')
            p_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make the dataset.')
    parser.add_argument('skill_name', type=str, help='name of the skill', default="push")
    parser.add_argument('perception_dir', type=str, help='path to perception directory', default="./dump/perception/Stow_push/raw/16-Apr-2023-12:09:05.613958")
    args = parser.parse_args()


    main(args.skill_name, args.perception_dir)
