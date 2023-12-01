import os
import torch
import random
import numpy as np
from ray.rllib.models.torch.torch_action_dist import TorchBeta, TorchDiagGaussian

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def extract_model_data(data, target_names, cameras, device):
    '''Extract the data that the model needs and the targets'''
    src_images = [[(data['current'][i][camera_type]).to(device) for camera_type in cameras] \
        for i in range(len(data['current']))]
    src_directions = [torch.stack(data['current'][i]['can_bus']['direction'], 1).float().to(device) \
        for i in range(len(data['current']))]
    src_speed = [(data['current'][i]['can_bus']['speed']).float().view(-1, 1).to(device) \
        for i in range(len(data['current']))]
    target = [torch.stack([data['current'][i]['can_bus'][x] for x in target_names], 1).float().squeeze().to(device) for i in range(len(data['current']))]
    return src_images, src_directions, src_speed, target

def extract_model_data_tensors(data, target_names, cameras, device):
    '''Extract the data that the model needs and the targets'''
    src_images = torch.stack([(data['current'][0][camera_type]).to(device) for camera_type in cameras], \
                             dim=1).unsqueeze(1).to(device)
    src_directions = torch.stack(data['current'][0]['can_bus']['direction'], 1).float().to(device)
    src_speed = (data['current'][0]['can_bus']['speed']).float().view(-1, 1).to(device)
    target = torch.stack([data['current'][0]['can_bus'][x] for x in target_names], 1).float().to(device)

    return src_images, src_directions, src_speed, target

def extract_model_data_tensors_no_device(data, target_names, cameras):
    '''Extract the data that the model needs and the targets'''
    src_images = torch.stack([(data['current'][0][camera_type]) for camera_type in cameras], \
                             dim=1).unsqueeze(1)
    src_directions = torch.stack(data['current'][0]['can_bus']['direction'], 1).float()
    src_speed = (data['current'][0]['can_bus']['speed']).float().view(-1, 1)
    target = torch.stack([data['current'][0]['can_bus'][x] for x in target_names], 1).float()

    return src_images, src_directions, src_speed, target

def forward_actor_critic(model, data, use_gaussian=False):
    if use_gaussian:
        diag_out = model.forward(*data)
        dist = TorchDiagGaussian(diag_out, None)
        out = torch.clip(dist.sample().squeeze(1), -1, 1)
        return out[:, 0], out[:, 1]
    else:
        beta_out = model.forward(*data)
        dist = TorchBeta(beta_out, None, low=-1, high=1)
        out = dist.sample().squeeze(1)
        return out[:, 0], out[:, 1]

def to_native_path(path: str) -> str:
    '''Changes the "/" character to the native path separator'''
    return os.path.join(*path.split('/'))

def get_config_path(config: str) -> str:
    '''Append the './train/configs' to the path if needed'''
    config = to_native_path(config)
    if os.path.sep in config:
        return config
    else:
        return os.path.join(to_native_path('./train/configs'), config)

def update_to_abspath(conf, fields):
    for field in fields:
        value = conf.get(field, None)
        if value is not None:
            value = os.path.abspath(to_native_path(value))
            conf.update({field: value})

