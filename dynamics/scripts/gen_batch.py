import itertools

# Define hyperparameters
hyperparameters = {
    "loss_type": ['mse', 'chamfer_emd'],
    "neighbor_radius": [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25],
    "tool_neighbor_radius": [0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25],
    "pstep": [2],
    "check_tool_touching": [0],
    "tool_next_edge": [0],
    "dynamic_staic_edge": [0, 1]
}

# Define default parameters
default_parameters = {
    "loss_type": 'mse',
    "neighbor_radius": 0.15,
    "tool_neighbor_radius": 0.15,
    "pstep": 2,
    "check_tool_touching": 0,
    "tool_next_edge": 0,
    "dynamic_staic_edge": 1,
}

# Define other parameters
other_parameters = {
    'stage': 'dy',
    'data_type': 'gt',
    'debug': 0,
    'n_his': 1,
    'sequence_length': 1,
    'time_gap': 1,
    'rigid_motion': 1,
    'attn': 0,
    'chamfer_weight': 0.5,
    'emd_weight': 0.5,
    'eval': 1
}

# Define skill names
skill_names = ['sweep', 'insert', 'push']

# Generate bash script
i = 0
for skill_name in skill_names:
    for hyperparameter, values in hyperparameters.items():
        for value in values:
            parameters = default_parameters.copy()
            parameters.update(other_parameters)
            parameters[hyperparameter] = value
            parameters['skill_name'] = skill_name  # add the current skill name to the parameters
            with open(f'dynamics/scripts/swbatch/run_batch_{i:03d}.swb', 'w') as f:
                    f.write('#!/bin/bash\n')
                    f.write('#SBATCH --job-name="stow"\n')
                    f.write('#SBATCH --partition=gpux1\n')
                    f.write('#SBATCH --time=24\n')
                    f.write('#SBATCH --output=../"../../hal_output/stow.%j.%N.out"\n')
                    f.write('#SBATCH --error="../../../hal_output/stow.%j.%N.err"\n\n')

                    f.write('module load opence\n')
                    f.write('conda activate my_opence\n\n')

                    f.write('python ../../train.py \\\n')
                    for key, value in parameters.items():
                        f.write(f'\t--{key} {value} \\\n')
                    f.write('\n')
            i = i + 1
            
