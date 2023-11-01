
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np

from configs import g_conf



def canbus_normalization(can_bus_dict, data_ranges):
    for type in data_ranges.keys():
        # we normalize the steering in range of [-1.0, 1.0]
        if type in ['steer', 'acceleration']:
            if not data_ranges[type] == [-1.0, 1.0]:
                print('')
                print('normalizing data for ', type)
                print('')
                # we normalize steering data if they are not from [-1.0, 1.0]
                can_bus_dict[type] = 2 *((can_bus_dict[type] - data_ranges[type][0]) / (data_ranges[type][1] - data_ranges[type][0])) - 1   # [-1.0, 1.0]
        elif type in ['throttle', 'brake', 'speed']:
            # we normalize the other can bus data in range of [0.0, 1.0]
            if not data_ranges[type] == [0.0, 1.0]:
                can_bus_dict[type] = abs(can_bus_dict[type]-data_ranges[type][0])/(data_ranges[type][1]-data_ranges[type][0])     # [0.0, 1.0]

        else:
            raise KeyError('The transformation of this data type has not yet defined:'+type)

    if 'direction' in can_bus_dict.keys():
        # we encode directions to one-hot vector
        if g_conf.DATA_COMMAND_ONE_HOT:
            if g_conf.DATA_COMMAND_CLASS_NUM == 4:
                can_bus_dict['direction'] = encode_directions_4(can_bus_dict['direction'])
            elif g_conf.DATA_COMMAND_CLASS_NUM == 6:
                can_bus_dict['direction'] = encode_directions_6(can_bus_dict['direction'])
        else:
            # we remark directions from 1-4 to 0-3 for torch.embedding layer
            can_bus_dict['direction'] = [can_bus_dict['direction']-1]
    return can_bus_dict

def train_transform(data, image_shape, augmentation=False):
    """
        Apply transformations and augmentations. The
        output is from 0-1 float.
    """
    if augmentation:
        pass

    else:
        for camera_type in g_conf.DATA_USED:
            image = data[camera_type]
            ## WE ALREADY PRE-PROCESSED IMAGES TO DESIRED SIZE
            # height = image_shape[1]
            # width = image_shape[2]
            # image = image.resize((width, height))
            image = TF.to_tensor(image)
            image = TF.normalize(image, mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
            data[camera_type] = image

    return data

def val_transform(data, image_shape):
    for camera_type in g_conf.DATA_USED:
        image = data[camera_type]
        ## WE ALREADY PRE-PROCESSED IMAGES TO DESIRED SIZE
        # height = image_shape[1]
        # width = image_shape[2]
        # image = image.resize((width, height))
        image = TF.to_tensor(image)
        image = TF.normalize(image,  mean=g_conf.IMG_NORMALIZATION['mean'], std=g_conf.IMG_NORMALIZATION['std'])
        data[camera_type] = image
    return data


def encode_directions_6(directions):
    # TURN_LEFT
    if float(directions) == 1.0:
        return [1, 0, 0, 0, 0, 0]
    # TURN_RIGHT
    elif float(directions) == 2.0:
        return [0, 1, 0, 0, 0, 0]
    # GO_STRAIGHT
    elif float(directions) == 3.0:
        return [0, 0, 1, 0, 0, 0]
    # FOLLOW_LANE
    elif float(directions) == 4.0:
        return [0, 0, 0, 1, 0, 0]
    # CHANGELANE_LEFT
    elif float(directions) == 5.0:
        return [0, 0, 0, 0, 1, 0]
    # CHANGELANE_RIGHT
    elif float(directions) == 6.0:
        return [0, 0, 0, 0, 0, 1]
    else:
        raise ValueError("Unexpcted direction identified %s" % str(directions))

def encode_directions_4(directions):
    # TURN_LEFT
    if float(directions) == 1.0:
        return [1, 0, 0, 0]
    # TURN_RIGHT
    elif float(directions) == 2.0:
        return [0, 1, 0, 0]
    # GO_STRAIGHT
    elif float(directions) == 3.0:
        return [0, 0, 1, 0]
    # FOLLOW_LANE
    elif float(directions) == 4.0:
        return [0, 0, 0, 1]
    else:
        raise ValueError("Unexpcted direction identified %s" % str(directions))

def decode_directions_6(one_hot_direction):
    one_hot_direction = list(one_hot_direction)
    index = one_hot_direction.index(max(one_hot_direction))
    # TURN_LEFT
    if index == 0:
        return 1.0
    # TURN_RIGHT
    elif index == 1:
        return 2.0
    # GO_STRAIGHT
    elif index == 2:
        return 3.0
    # FOLLOW_LANE
    elif index == 3:
        return 4.0
    # CHANGELANE_LEFT
    elif index == 4:
        return 5.0
    # CHANGELANE_RIGHT
    elif index == 5:
        return 6.0
    else:
        raise ValueError("Unexpcted direction identified %s" % one_hot_direction)

def decode_directions_4(one_hot_direction):
    one_hot_direction = list(one_hot_direction)
    index = one_hot_direction.index(max(one_hot_direction))
    # TURN_LEFT
    if index == 0:
        return 1.0
    # TURN_RIGHT
    elif index == 1:
        return 2.0
    # GO_STRAIGHT
    elif index == 2:
        return 3.0
    # FOLLOW_LANE
    elif index == 3:
        return 4.0
    else:
        raise ValueError("Unexpcted direction identified %s" % one_hot_direction)

def inverse_normalize_(tensor, mean, std):
    tensors=[]
    for i in range(tensor.shape[0]):
        img = torch.clamp(tensor[i] * torch.tensor((std)).view(3, 1, 1).to(tensor.get_device())+torch.tensor((mean)).view(3, 1, 1).to(tensor.get_device()), 0, 1)
        tensors.append(img)
    return torch.stack(tensors, dim=0)


def inverse_normalize(tensor, mean, std):

    inv_normalize = transforms.Normalize(
        mean=[-mean[0]/std[0], -mean[1]/std[1], -mean[2]/std[2]],
        std=[1/std[0], 1/std[1], 1/std[2]])

    return inv_normalize(tensor)