import torch.nn as nn


class Mu_Logvar(nn.Module):

    def __init__(self, bottleneck_layers= None):

        super(Mu_Logvar, self).__init__()

        if bottleneck_layers is None:
            raise ValueError("No bottleneck layers provided")

        self.bottleneck_layers = nn.ModuleList(bottleneck_layers)

    def forward(self, x):

        outputs = []
        for layer in self.bottleneck_layers:
            outputs.append(layer(x))
        return outputs




