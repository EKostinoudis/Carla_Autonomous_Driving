"""
    A simple factory module that returns instances of possible modules 

"""

from .models import CILv2_multiview_attention


def Models(architecture_name, configuration):

    # Baseline end-to-end behavior cloning model, with TFM in multi-view feature space
    if architecture_name == 'CILv2_multiview_attention':
        return CILv2_multiview_attention(configuration)
    else:
        raise NotImplementedError(" Not found architecture name")
