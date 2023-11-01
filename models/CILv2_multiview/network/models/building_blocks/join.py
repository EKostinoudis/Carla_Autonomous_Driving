
import torch.nn as nn
import torch


class Join(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        # TODO:  For now the end module is a case
        # TODO: Make an auto naming function for this.

        super(Join, self).__init__()

        self.norm = None

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'mode' not in params:
            raise ValueError(" Missing the mode parameter ")
        if 'after_process' not in params:
            raise ValueError(" Missing the after_process parameter ")

        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization

        self.after_process = params['after_process']
        self.mode = params['mode']

    def forward(self, x, m, c=None):
        # get only the speeds from measurement labels

        if self.mode == 'cat':
            if c is None:
                j = torch.cat((x, m), 1)
            else:
                j = torch.cat((x, m, c), 1)

        else:
            raise ValueError("Mode to join networks not found")

        return self.after_process(j)

    def forward_list(self, x_list):
        # get only the speeds from measurement labels

        if self.mode == 'cat':
            j = torch.cat(x_list, 1)

        else:
            raise ValueError("Mode to join networks not found")

        return self.after_process(j)



