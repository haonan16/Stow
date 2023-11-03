#!/bin/bash

if [[ $# -lt 1 ]]; then
    echo "Usage: ./run_sample.sh <num_rollouts> [--skill-name=<name1> --skill-name=<name2> ...]"
    exit 1
fi

NUM_ROLLOUTS=$1
shift

# Parse named arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --skill-name=*)
        SKILL_NAMES+=("${key#*=}")
        shift
        ;;
        *)
        echo "Unknown named argument: $key"
        exit 1
        ;;
    esac
done

# Use default skill names if not provided
if [[ ${#SKILL_NAMES[@]} -eq 0 ]]; then
    SKILL_NAMES=("push" "sweep" "insert")
fi

# Loop through skill names and run sample.py for each skill
for skill_name in "${SKILL_NAMES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 \
    python perception/sample.py \
        --stage perception \
        --num_workers 10 \
        --n_rollouts $NUM_ROLLOUTS \
        --skill_name $skill_name &
done

wait


# num_workers = 1
# n_rollouts = 100

# CUDA_VISIBLE_DEVICES=0		\
# python perception/sample.py \
# 	--stage perception \
# 	--num_workers $num_workers \
# 	--n_rollouts $n_rollouts \
# 	--skill_name push &

# CUDA_VISIBLE_DEVICES=0		\
# python perception/sample.py \
# 	--stage perception \
# 	--num_workers $num_workers \
# 	--n_rollouts $n_rollouts \
# 	--skill_name sweep &

# CUDA_VISIBLE_DEVICES=0		\
# python perception/sample.py \
# 	--stage perception \
# 	--num_workers $num_workers \
# 	--n_rollouts $n_rollouts \
# 	--skill_name insert &

# wait