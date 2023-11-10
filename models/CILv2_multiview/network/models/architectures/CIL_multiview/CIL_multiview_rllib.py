import importlib

# from configs import g_conf
from network.models.building_blocks import FC
from network.models.building_blocks.PositionalEncoding import PositionalEncoding
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch
import torch.nn as nn

class CIL_multiview_actor_critic(nn.Module):
    def __init__(self, g_conf):
        nn.Module.__init__(self)

        self.g_conf = g_conf
        self.params = g_conf['MODEL_CONFIGURATION']

        resnet_module = importlib.import_module('network.models.building_blocks.resnet_FM')
        resnet_module = getattr(resnet_module, self.params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = resnet_module(pretrained=g_conf['IMAGENET_PRE_TRAINED'],
                                                          layer_id = self.params['encoder_embedding']['perception']['res']['layer_id'])
        _, self.res_out_dim, self.res_out_h, self.res_out_w = self.encoder_embedding_perception.get_backbone_output_shape([g_conf['BATCH_SIZE']] + g_conf['IMAGE_SHAPE'])[self.params['encoder_embedding']['perception']['res'][ 'layer_id']]

        if self.params['TxEncoder']['learnable_pe']:
            self.positional_encoding = nn.Parameter(torch.zeros(1, len(g_conf['DATA_USED'])*g_conf['ENCODER_INPUT_FRAMES_NUM']*self.res_out_h*self.res_out_w, self.params['TxEncoder']['d_model']))
        else:
            self.positional_encoding = PositionalEncoding(d_model=self.params['TxEncoder']['d_model'], dropout=0.0, max_len=len(g_conf['DATA_USED'])*g_conf['ENCODER_INPUT_FRAMES_NUM']*self.res_out_h*self.res_out_w)

        join_dim = self.params['TxEncoder']['d_model']
        self.command = nn.Linear(g_conf['DATA_COMMAND_CLASS_NUM'], self.params['TxEncoder']['d_model'])
        self.speed = nn.Linear(1, self.params['TxEncoder']['d_model'])

        tx_encoder_layer = TransformerEncoderLayer(d_model=self.params['TxEncoder']['d_model'],
                                                   nhead=self.params['TxEncoder']['n_head'],
                                                   norm_first=self.params['TxEncoder']['norm_first'], batch_first=True)
        self.tx_encoder = TransformerEncoder(tx_encoder_layer, num_layers=self.params['TxEncoder']['num_layers'],
                                             norm=nn.LayerNorm(self.params['TxEncoder']['d_model']))

        # actor output (action)
        self.action_output = FC(params={'neurons': [join_dim] +
                                            self.params['action_output']['fc']['neurons'] +
                                            [4],
                                 'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        # critic output (value)
        self.value_output = FC(params={'neurons': [join_dim] +
                                            self.params['action_output']['fc']['neurons'] +
                                            [1],
                                 'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        self.train()

    def value_function(self):
        return self._value_out.view(-1)

    def forward(self, s, s_d, s_s):
        S = int(self.g_conf['ENCODER_INPUT_FRAMES_NUM'])
        B = s_d.shape[0]

        x = s
        # x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1) # [B, S, cam, 3, H, W]
        x = x.view(B*S*len(self.g_conf['DATA_USED']), self.g_conf['IMAGE_SHAPE'][0], self.g_conf['IMAGE_SHAPE'][1], self.g_conf['IMAGE_SHAPE'][2])  # [B*S*cam, 3, H, W]
        d = s_d
        s = s_s

        # image embedding
        e_p, _ = self.encoder_embedding_perception(x)    # [B*S*cam, dim, h, w]
        encoded_obs = e_p.view(B, S*len(self.g_conf['DATA_USED']), self.res_out_dim, self.res_out_h*self.res_out_w)  # [B, S*cam, dim, h*w]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.res_out_dim)  # [B, S*cam*h*w, 512]
        e_d = self.command(d).unsqueeze(1)     # [B, 1, 512]
        e_s = self.speed(s).unsqueeze(1)       # [B, 1, 512]

        encoded_obs = encoded_obs + e_d + e_s

        if self.params['TxEncoder']['learnable_pe']:
            # positional encoding
            pe = encoded_obs + self.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer encoder multi-head self-attention layers
        in_memory, _ = self.tx_encoder(pe)  # [B, S*cam*h*w, 512]

        in_memory = torch.mean(in_memory, dim=1)  # [B, 512]

        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, 512) -> (B, 1, len(TARGETS))

        return action_output         # (B, 1, 1), (B, 1, len(TARGETS))


class CIL_multiview_actor_critic_stack(nn.Module):
    def __init__(self, g_conf):
        nn.Module.__init__(self)

        self.g_conf = g_conf
        self.params = g_conf['MODEL_CONFIGURATION']

        resnet_module = importlib.import_module('network.models.building_blocks.resnet_FM')
        resnet_module = getattr(resnet_module, self.params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = resnet_module(pretrained=g_conf['IMAGENET_PRE_TRAINED'],
                                                          layer_id = self.params['encoder_embedding']['perception']['res']['layer_id'])
        _, self.res_out_dim, self.res_out_h, self.res_out_w = self.encoder_embedding_perception.get_backbone_output_shape([g_conf['BATCH_SIZE']] + g_conf['IMAGE_SHAPE'])[self.params['encoder_embedding']['perception']['res'][ 'layer_id']]

        if self.params['TxEncoder']['learnable_pe']:
            self.positional_encoding = nn.Parameter(torch.zeros(1, len(g_conf['DATA_USED'])*g_conf['ENCODER_INPUT_FRAMES_NUM']*self.res_out_h*self.res_out_w, self.params['TxEncoder']['d_model']))
        else:
            self.positional_encoding = PositionalEncoding(d_model=self.params['TxEncoder']['d_model'], dropout=0.0, max_len=len(g_conf['DATA_USED'])*g_conf['ENCODER_INPUT_FRAMES_NUM']*self.res_out_h*self.res_out_w)

        join_dim = self.params['TxEncoder']['d_model']
        self.command = nn.Linear(g_conf['DATA_COMMAND_CLASS_NUM'], self.params['TxEncoder']['d_model'])
        self.speed = nn.Linear(1, self.params['TxEncoder']['d_model'])

        tx_encoder_layer = TransformerEncoderLayer(d_model=self.params['TxEncoder']['d_model'],
                                                   nhead=self.params['TxEncoder']['n_head'],
                                                   norm_first=self.params['TxEncoder']['norm_first'], batch_first=True)
        self.tx_encoder = TransformerEncoder(tx_encoder_layer, num_layers=self.params['TxEncoder']['num_layers'],
                                             norm=nn.LayerNorm(self.params['TxEncoder']['d_model']))

        # actor output (action)
        self.action_output = FC(params={'neurons': [join_dim] +
                                            self.params['action_output']['fc']['neurons'] +
                                            [2],
                                 'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        self.action_output2 = FC(params={'neurons': [join_dim] +
                                            [self.params['action_output']['fc']['neurons'][-1] + 2] +
                                            [128, 4],
                                 'dropouts': 2*[0.0],
                                 'end_layer': True})

        # critic output (value)
        self.value_output = FC(params={'neurons': self.params['action_output']['fc']['neurons'] + [1],
                                 'dropouts': self.params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        self.train()

    def value_function(self):
        return self._value_out.view(-1)

    def forward(self, s, s_d, s_s):
        S = int(self.g_conf['ENCODER_INPUT_FRAMES_NUM'])
        B = s_d.shape[0]

        x = s
        # x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1) # [B, S, cam, 3, H, W]
        x = x.view(B*S*len(self.g_conf['DATA_USED']), self.g_conf['IMAGE_SHAPE'][0], self.g_conf['IMAGE_SHAPE'][1], self.g_conf['IMAGE_SHAPE'][2])  # [B*S*cam, 3, H, W]
        d = s_d
        s = s_s

        # image embedding
        e_p, _ = self.encoder_embedding_perception(x)    # [B*S*cam, dim, h, w]
        encoded_obs = e_p.view(B, S*len(self.g_conf['DATA_USED']), self.res_out_dim, self.res_out_h*self.res_out_w)  # [B, S*cam, dim, h*w]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.res_out_dim)  # [B, S*cam*h*w, 512]
        e_d = self.command(d).unsqueeze(1)     # [B, 1, 512]
        e_s = self.speed(s).unsqueeze(1)       # [B, 1, 512]

        encoded_obs = encoded_obs + e_d + e_s

        if self.params['TxEncoder']['learnable_pe']:
            # positional encoding
            pe = encoded_obs + self.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer encoder multi-head self-attention layers
        in_memory, _ = self.tx_encoder(pe)  # [B, S*cam*h*w, 512]

        in_memory = torch.mean(in_memory, dim=1)  # [B, 512]

        # action_output = self.action_output(in_memory).unsqueeze(1)  # (B, 512) -> (B, 1, len(TARGETS))
        x = in_memory
        for action_layer in self.action_output.layers[:-1]:
            x = action_layer(x)  # (B, 256)
        old_out = self.action_output.layers[-1](x)  # (B, 2)

        action_output = self.action_output2(torch.cat((x, old_out), dim=1)).unsqueeze(1)  # (B, 514) -> (B, 1, len(TARGETS))

        return action_output         # (B, 1, 1), (B, 1, len(TARGETS))

class CIL_multiview_rllib(TorchModelV2, CIL_multiview_actor_critic):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, g_conf):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CIL_multiview_actor_critic.__init__(self, g_conf)

    def forward(self, input_dict, state, seq_lens):
        s, s_d, s_s = input_dict["obs"]
        action_output = CIL_multiview_actor_critic.forward(self, s, s_d, s_s)
        return action_output, state

class CIL_multiview_rllib_stack(TorchModelV2, CIL_multiview_actor_critic_stack):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, g_conf):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        CIL_multiview_actor_critic_stack.__init__(self, g_conf)

    def forward(self, input_dict, state, seq_lens):
        s, s_d, s_s = input_dict["obs"]
        action_output = CIL_multiview_actor_critic_stack.forward(self, s, s_d, s_s)
        return action_output, state
