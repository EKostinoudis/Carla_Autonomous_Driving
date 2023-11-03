#!/bin/bash

# name of script to run with accelerate
script="policy_head_training_accelerate.py"

# Get the current directory name
current_directory=$(basename "$(pwd)")

# Check if the directory name contains the word "train"
if [[ $current_directory == "train" ]]; then
	folder="."
else
	folder="train"
fi

path="$folder/$script"

if [ -f "$path" ]; then
	python -m accelerate.commands.launch \
		--multi_gpu \
		--num_processes 0 \
		--num_machines 1 \
		--mixed_precision no \
		--dynamo_backend no \
		$path --clean
else
	echo $script not found
fi

