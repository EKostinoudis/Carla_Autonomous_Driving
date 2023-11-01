from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from configs.attribute_dict import AttributeDict
import copy
import numpy as np
import os
import yaml

# from logger._logger import create_log

_g_conf = AttributeDict()

_g_conf.immutable(False)

"""#### GENERAL CONFIGURATION PARAMETERS ####"""
"""#### Training Related Parameters"""
_g_conf.MAGICAL_SEED = 0
_g_conf.NUM_WORKER = 14
_g_conf.DATA_PARALLEL = False
_g_conf.TRAINING_RESUME = True
_g_conf.BATCH_SIZE = 120
_g_conf.NUMBER_EPOCH = 100     # Total number of training iteration
_g_conf.TRAIN_DATASET_NAME = []
_g_conf.VALID_DATASET_NAME = []      # More than one datasets could be evaluated, thus a list
_g_conf.DATA_USED = ['rgb_left', 'rgb_central', 'rgb_right']
_g_conf.IMAGE_SHAPE = [3, 88, 200]
_g_conf.ENCODER_INPUT_FRAMES_NUM = 1
_g_conf.ENCODER_STEP_INTERVAL = 1     # the pace step of frame you want to use. For example, if you want to have 5 sequential input images taking pre 20-frames as a step, you should set INPUT_FRAMES_NUM =5 and INPUT_FRAME_INTERVAL=20
_g_conf.ENCODER_OUTPUT_STEP_DELAY = 0  # whether we want to predict the future data points or just the current point
_g_conf.DECODER_OUTPUT_FRAMES_NUM= 1
_g_conf.AUGMENTATION = False
_g_conf.DATA_FPS = 10
_g_conf.DATA_COMMAND_CLASS_NUM = 4
_g_conf.DATA_COMMAND_ONE_HOT = True
_g_conf.DATA_NORMALIZATION = {}
_g_conf.IMG_NORMALIZATION = {'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}   # ImageNet by default
_g_conf.EXP_SAVE_PATH = '_results'
_g_conf.TARGETS = ['steer', 'throttle', 'brake']  # From the float data, the ones that the network should estimate
_g_conf.ACCELERATION_AS_ACTION = False
_g_conf.OTHER_INPUTS= ['speed'] # From the float data, the ones that are input to the neural network

"""#### Optimizer Related Parameters ####"""
_g_conf.LOSS = ''    # It can be the name of loss, such as L1, CrossEntropy, or an architecure name such as "fasterRcnn, deeplabv3", which means we use the same loss as this architectures
_g_conf.LOSS_WEIGHT = {}
_g_conf.LEARNING_RATE = 0.0002       # the original learning rate setting
_g_conf.LEARNING_RATE_DECAY = True
_g_conf.LEARNING_RATE_MINIMUM = 0.00001
_g_conf.LEARNING_RATE_DECAY_EPOCHES = []    # we adjust learning rate for each 1000 iterations
_g_conf.LEARNING_RATE_POLICY = {'name': 'normal', 'level': 0.5, 'momentum': 0, 'weight_decay': 0}   # lr multiply by 0.5 for each LEARNING_RATE_STEP

"""#### Network Related Parameters ####"""
_g_conf.MODEL_TYPE = ''
_g_conf.MODEL_CONFIGURATION = {}
_g_conf.IMAGENET_PRE_TRAINED = False
_g_conf.LOAD_CHECKPOINT = ''

"""#### Validation Related Parameters"""
_g_conf.EVAL_SAVE_LAST_Conv_ACTIVATIONS = True     # the last Conv. layer of backbone that to be saved attention maps
_g_conf.EVAL_BATCH_SIZE = 1          # batch size for evaluation
_g_conf.EVAL_SAVE_EPOCHES = [1]      # we specifize the epoch we want to do offline evaluation
_g_conf.EVAL_IMAGE_WRITING_NUMBER = 10
_g_conf.EARLY_STOPPING = False          # By default, we do not apply early stopping
_g_conf.EARLY_STOPPING_PATIENCE = 3
_g_conf.EVAL_DRAW_OFFLINE_RESULTS_GRAPHS = ['MAE']

"""#### Logger Related Parameters"""
_g_conf.TRAIN_LOG_SCALAR_WRITING_FREQUENCY = 2
_g_conf.TRAIN_IMAGE_WRITING_NUMBER = 2
_g_conf.TRAIN_IMAGE_LOG_FREQUENCY = 1000
_g_conf.TRAIN_PRINT_LOG_FREQUENCY = 100

def merge_with_yaml(yaml_filename, process_type='train_val'):
    """Load a yaml config file and merge it into the global config object"""
    global _g_conf
    with open(yaml_filename, 'r') as f:

        # yaml_file = yaml.load(f)
        yaml_file = yaml.safe_load(f)

        yaml_cfg = AttributeDict(yaml_file)

    path_parts = os.path.split(yaml_filename)
    if process_type == 'train_val':
        _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
        _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    if process_type == 'val_only':
        _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(path_parts[-2])[-1]
        _g_conf.EXPERIMENT_NAME = path_parts[-1].split('.')[-2]
    elif process_type == 'drive':
        _g_conf.EXPERIMENT_BATCH_NAME = os.path.split(os.path.split(path_parts[-2])[-2])[-1]
        _g_conf.EXPERIMENT_NAME = os.path.split(path_parts[-2])[-1]
    _merge_a_into_b(yaml_cfg, _g_conf)


def get_names(folder):
    alias_in_folder = os.listdir(os.path.join('configs', folder))

    experiments_in_folder = {}
    for experiment_alias in alias_in_folder:

        g_conf.immutable(False)
        merge_with_yaml(os.path.join('configs', folder, experiment_alias))

        experiments_in_folder.update({experiment_alias: g_conf.EXPERIMENT_GENERATED_NAME})

    return experiments_in_folder


def create_exp_path(root, experiment_batch, experiment_name):
    # This is hardcoded the logs always stay on the _logs folder
    root_path = os.path.join(root, '_results')
    if not os.path.exists(os.path.join(root_path, experiment_batch, experiment_name)):
        os.makedirs(os.path.join(root_path, experiment_batch, experiment_name))


def set_type_of_process(process_type, root):
    """
    This function is used to set which is the type of the current process, test, train or val
    and also the details of each since there could be many vals and tests for a single
    experiment.

    NOTE: AFTER CALLING THIS FUNCTION, THE CONFIGURATION CLOSES

    Args:
        type:

    Returns:

    """

    if process_type == 'train_val' or process_type == 'train_only' or process_type == 'val_only' or process_type == 'drive':
        _g_conf.PROCESS_NAME = process_type
        if not os.path.exists(os.path.join(root,'_results', _g_conf.EXPERIMENT_BATCH_NAME,
                                           _g_conf.EXPERIMENT_NAME,
                                           'checkpoints')):
            os.mkdir(os.path.join(root,'_results', _g_conf.EXPERIMENT_BATCH_NAME,
                                  _g_conf.EXPERIMENT_NAME,
                                  'checkpoints'))

        _g_conf.EXP_SAVE_PATH = os.path.join(root,'_results', _g_conf.EXPERIMENT_BATCH_NAME, _g_conf.EXPERIMENT_NAME)

    else:
        raise ValueError("Not found type of process")

    '''
    if process_type == 'train_val' or process_type == 'train_only' or process_type == 'val_only':
        create_log(_g_conf.EXP_SAVE_PATH,
                _g_conf.TRAIN_LOG_SCALAR_WRITING_FREQUENCY,
                _g_conf.TRAIN_IMAGE_LOG_FREQUENCY)
    '''


    _g_conf.immutable(True)


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """

    assert isinstance(a, AttributeDict) or isinstance(a, dict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttributeDict) or isinstance(a, dict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if is it more than second stack
            if stack is not None:
                b[k] = v_
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)

        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts

        b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects


    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #

    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, type(None)):
        value_a = value_a
    elif isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_b, range) and not isinstance(value_a, list):
        value_a = eval(value_a)
    elif isinstance(value_b, range) and isinstance(value_a, list):
        value_a = list(value_a)
    elif isinstance(value_b, dict):
        value_a = eval(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


g_conf = _g_conf

