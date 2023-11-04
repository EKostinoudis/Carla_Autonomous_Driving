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

# get an unused port
port=$(python -c "import socket; s = socket.socket(); s.bind(('', 0));print(s.getsockname()[1]);s.close()")

if [ -f "$path" ]; then
	$python_cmd -m accelerate.commands.launch \
		--main_process_port $port \
		--multi_gpu \
		--num_machines 1 \
		--mixed_precision no \
		--dynamo_backend no \
		$path "$@"
else
	echo $script not found
fi

