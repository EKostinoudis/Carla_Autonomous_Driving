import torch
import torch.nn as nn
import importlib

from configs import g_conf
from network.models.building_blocks import FC
from network.models.building_blocks.PositionalEncoding import PositionalEncoding
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoder
from network.models.building_blocks.Transformer.TransformerEncoder import TransformerEncoderLayer


class CIL_multiview(nn.Module):
    def __init__(self, params):
        super(CIL_multiview, self).__init__()
        self.params = params

        resnet_module = importlib.import_module('network.models.building_blocks.resnet_FM')
        resnet_module = getattr(resnet_module, params['encoder_embedding']['perception']['res']['name'])
        self.encoder_embedding_perception = resnet_module(pretrained=g_conf.IMAGENET_PRE_TRAINED,
                                                          layer_id = params['encoder_embedding']['perception']['res'][ 'layer_id'])
        _, self.res_out_dim, self.res_out_h, self.res_out_w = self.encoder_embedding_perception.get_backbone_output_shape([g_conf.BATCH_SIZE] + g_conf.IMAGE_SHAPE)[params['encoder_embedding']['perception']['res'][ 'layer_id']]

        if params['TxEncoder']['learnable_pe']:
            self.positional_encoding = nn.Parameter(torch.zeros(1, len(g_conf.DATA_USED)*g_conf.ENCODER_INPUT_FRAMES_NUM*self.res_out_h*self.res_out_w, params['TxEncoder']['d_model']))
        else:
            self.positional_encoding = PositionalEncoding(d_model=params['TxEncoder']['d_model'], dropout=0.0, max_len=len(g_conf.DATA_USED)*g_conf.ENCODER_INPUT_FRAMES_NUM*self.res_out_h*self.res_out_w)

        join_dim = params['TxEncoder']['d_model']
        self.command = nn.Linear(g_conf.DATA_COMMAND_CLASS_NUM, params['TxEncoder']['d_model'])
        self.speed = nn.Linear(1, params['TxEncoder']['d_model'])

        tx_encoder_layer = TransformerEncoderLayer(d_model=params['TxEncoder']['d_model'],
                                                   nhead=params['TxEncoder']['n_head'],
                                                   norm_first=params['TxEncoder']['norm_first'], batch_first=True)
        self.tx_encoder = TransformerEncoder(tx_encoder_layer, num_layers=params['TxEncoder']['num_layers'],
                                             norm=nn.LayerNorm(params['TxEncoder']['d_model']))

        self.action_output = FC(params={'neurons': [join_dim] +
                                            params['action_output']['fc']['neurons'] +
                                            [len(g_conf.TARGETS)],
                                 'dropouts': params['action_output']['fc']['dropouts'] + [0.0],
                                 'end_layer': True})

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

        self.train()

    def forward(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1) # [B, S, cam, 3, H, W]
        x = x.view(B*S*len(g_conf.DATA_USED), g_conf.IMAGE_SHAPE[0], g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        e_p, _ = self.encoder_embedding_perception(x)    # [B*S*cam, dim, h, w]
        encoded_obs = e_p.view(B, S*len(g_conf.DATA_USED), self.res_out_dim, self.res_out_h*self.res_out_w)  # [B, S*cam, dim, h*w]
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

    def foward_eval(self, s, s_d, s_s):
        S = int(g_conf.ENCODER_INPUT_FRAMES_NUM)
        B = s_d[0].shape[0]

        x = torch.stack([torch.stack(s[i], dim=1) for i in range(S)], dim=1)  # [B, S, cam, 3, H, W]
        x = x.view(B * S * len(g_conf.DATA_USED), g_conf.IMAGE_SHAPE[0], g_conf.IMAGE_SHAPE[1], g_conf.IMAGE_SHAPE[2])  # [B*S*cam, 3, H, W]
        d = s_d[-1]  # [B, 4]
        s = s_s[-1]  # [B, 1]

        # image embedding
        e_p, resnet_inter = self.encoder_embedding_perception(x)  # [B*S*cam, dim, h, w]
        encoded_obs = e_p.view(B, S * len(g_conf.DATA_USED), self.res_out_dim,  self.res_out_h * self.res_out_w)  # [B, S*cam, dim, h*w]
        encoded_obs = encoded_obs.transpose(2, 3).reshape(B, -1, self.res_out_dim)  # [B, S*cam*h*w, 512]
        e_d = self.command(d).unsqueeze(1)  # [B, 1, 512]
        e_s = self.speed(s).unsqueeze(1)  # [B, 1, 512]

        encoded_obs = encoded_obs + e_d + e_s   # [B, S*cam*h*w, 512]

        if self.params['TxEncoder']['learnable_pe']:
            # positional encoding
            pe = encoded_obs + self.positional_encoding    # [B, S*cam*h*w, 512]
        else:
            pe = self.positional_encoding(encoded_obs)

        # Transformer encoder multi-head self-attention layers
        in_memory, attn_weights = self.tx_encoder(pe)  # [B, S*cam*h*w, 512]
        in_memory = torch.mean(in_memory, dim=1)  # [B, 512]

        action_output = self.action_output(in_memory).unsqueeze(1)  # (B, 512) -> (B, 1, len(TARGETS))

        return action_output, resnet_inter, attn_weights

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

