
import torch.nn as nn


class FC_Bottleneck(nn.Module):
    def __init__(self, params=None, module_name='Default'):
        super(FC_Bottleneck, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'neurons' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")

        self.layers = []

        for i in range(0, len(params['neurons']) - 1):
            fc = nn.Linear(params['neurons'][i], params['neurons'][i + 1])

            self.layers.append(fc)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        # if X is a tuple, just return the other elements, the idea is to re pass
        # the intermediate layers for future attention plotting
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            return self.layers(x)



class FC(nn.Module):

    def __init__(self, params=None, norm=False, activate=True, module_name='Default'):
        # TODO: Make an auto naming function for this.

        super(FC, self).__init__()


        """" ---------------------- FC ----------------------- """
        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'neurons' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['neurons'])-1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")


        self.layers = []


        for i in range(0, len(params['neurons']) -1):

            fc = nn.Linear(params['neurons'][i], params['neurons'][i+1])
            dropout = nn.Dropout(p=params['dropouts'][i])
            #norm = nn.BatchNorm1d(params['neurons'][i+1])
            layernorm = nn.LayerNorm(params['neurons'][i+1], eps=1e-5)
            relu = nn.ReLU(inplace=True)

            if i == len(params['neurons'])-2 and params['end_layer']:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                if norm:
                    if activate:
                        #self.layers.append(nn.Sequential(*[fc, dropout, norm, relu]))
                        self.layers.append(nn.Sequential(*[fc, dropout, layernorm, relu]))
                    else:
                        self.layers.append(nn.Sequential(*[fc, dropout, layernorm]))
                else:
                    if activate:
                        self.layers.append(nn.Sequential(*[fc, dropout, relu]))
                    else:
                        self.layers.append(nn.Sequential(*[fc, dropout]))


        self.layers = nn.Sequential(*self.layers)



    def forward(self, x):# return_intermediate=None):
        # if X is a tuple, just return the other elements, the idea is to re pass
        # the intermediate layers for future attention plotting
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            #if return_intermediate is not None:

            #else:
            return self.layers(x)


