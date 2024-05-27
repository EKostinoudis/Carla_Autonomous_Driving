#!/bin/bash

port=$1

if [ "$#" -eq 2 ]; then
    device=$2
else
    device=$(( ((port - 2000) / 4) % 4 ))
fi

echo "Spawning server | port: $port | device: $device"
$HOME/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping \
    -quality-level=Epic \
    -RenderOffScreen \
    -nosound \
    -carla-rpc-port=$port \
    -graphicsadapter=$device

