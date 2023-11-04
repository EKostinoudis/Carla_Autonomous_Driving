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

if python --version 2>&1 | grep -q "Python 3"; then
	python_cmd="python"
else
	python_cmd="python3"
fi

if [ -f "$path" ]; then
	# use different port because there is a posibility for someone else to
	# use the default port (DistributedDataParallel default port)
	$python_cmd -m accelerate.commands.launch \
		--main_process_port 21359 \
		--multi_gpu \
		--num_machines 1 \
		--mixed_precision no \
		--dynamo_backend no \
		$path "$@"
else
	echo $script not found
fi

