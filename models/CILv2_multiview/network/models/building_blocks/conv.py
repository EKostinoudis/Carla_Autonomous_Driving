
import torch.nn as nn
import torch


class ConvTrans_Decode(nn.Module):
    def __init__(self, params=None, module_name='Default'):
        super(ConvTrans_Decode, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channels' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernels' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")

        self.layers = []

        for i in range(0, len(params['channels'])-1):
            conv_transpose = nn.ConvTranspose2d(in_channels=params['channels'][i], out_channels=params['channels'][i + 1],
                             kernel_size=params['kernels'][i], stride=params['strides'][i])
            relu = nn.ReLU(inplace=True)
            sigmoid = nn.Sigmoid()

            if i == len(params['channels'])-2 :
                layer = nn.Sequential(*[conv_transpose, sigmoid])
            else:
                layer = nn.Sequential(*[conv_transpose, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name

    def forward(self, x, h_shape):
        """ Each conv is: conv + relu or sigmoid"""
        x = x.view(x.shape[0], h_shape[1], h_shape[2], h_shape[3])
        x = self.layers(x)

        return x


class Conv_Encode(nn.Module):
    def __init__(self, params=None, module_name='Default'):
        super(Conv_Encode, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channels' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernels' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")

        self.layers = []

        for i in range(0, len(params['channels'])-1):
            conv = nn.Conv2d(in_channels=params['channels'][i], out_channels=params['channels'][i+1],
                             kernel_size=params['kernels'][i], stride=params['strides'][i])
            relu = nn.ReLU(inplace=True)

            layer = nn.Sequential(*[conv, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name


    def forward(self, x):

        """ Each conv is: conv + relu """
        x = self.layers(x)
        flatten = x.view(x.shape[0], -1)

        return flatten, x.shape

    def get_conv_output(self, shape):

        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat, _= self.forward(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size


class Conv(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        super(Conv, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channels' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernels' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['channels'])-1:
            raise ValueError("Dropouts should be from the len of channel_sizes minus 1")


        """" ------------------ IMAGE MODULE ---------------- """
        # Conv2d(input channel, output channel, kernel size, stride), Xavier initialization and 0.1 bias initialization


        self.layers = []

        for i in range(0, len(params['channels'])-1):
            conv = nn.Conv2d(in_channels=params['channels'][i], out_channels=params['channels'][i+1],
                             kernel_size=params['kernels'][i], stride=params['strides'][i])

            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(params['channels'][i+1])

            layer = nn.Sequential(*[conv, bn, dropout, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name


    def forward(self, x):

        """ Each conv is: conv + batch normalization + dropout + relu """
        x = self.layers(x)
        x = x.view(-1, self.num_flat_features(x))

        return x, self.layers


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def get_conv_output(self, shape):
        """
           By inputing the shape of the input, simulate what is the ouputsize.
        """

        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat, _ = self.forward(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

