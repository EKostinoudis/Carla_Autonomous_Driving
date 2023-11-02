import torch
from ray.rllib.models.torch.torch_action_dist import TorchBeta

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
    target = torch.stack([data['current'][0]['can_bus'][x] for x in target_names], 1).float().squeeze().to(device)

    return src_images, src_directions, src_speed, target

def forward_actor_critic(model, data):
    beta_out = model.forward2(*data)
    dist = TorchBeta(beta_out, None, low=-1, high=1)
    out = dist.sample().squeeze(1)
    return out[:, 0], out[:, 1]

