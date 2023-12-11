from typing import Mapping, Any
import torch

from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule

from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian

from .beta_distribution import TorchBetaDistribution
from .network.models.architectures.CIL_multiview.CIL_multiview_rllib import CIL_multiview_actor_critic_RLModule

class CILv2_RLModule(TorchRLModule, PPORLModule):
    def setup(self):
        self.model = CIL_multiview_actor_critic_RLModule(self.config.model_config_dict)

        # for now we use the model config for passing the distribution, the
        # proper way is the catalog class
        dist = self.config.model_config_dict.get('output_distribution', 'gaussian') 
        if dist == 'beta':
            self.action_dist_cls = TorchBetaDistribution
        elif dist == 'gaussian':
            self.action_dist_cls = TorchDiagGaussian
        else:
            raise ValueError(
                f'{dist} distribution not supported. Use "beta" or "gaussian"'
            )

    def get_initial_state(self) -> dict:
        return {}

    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}
        s, s_d, s_s = restore_original_dimensions(
            batch[SampleBatch.OBS],
            self.config.observation_space,
            tensorlib=torch
        )
        if len(s_d.shape) == 3: s_d = s_d.squeeze(1)
        if len(s_s.shape) == 3: s_s = s_s.squeeze(1)
        with torch.no_grad():
            action_logits = self.model.forward(s, s_d, s_s).squeeze(1)
        output[SampleBatch.VF_PREDS] = self.model._value_out.view(-1)
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        return output

    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._forward_inference(batch)

    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}
        s, s_d, s_s = restore_original_dimensions(
            batch[SampleBatch.OBS],
            self.config.observation_space,
            tensorlib=torch
        )
        if len(s_d.shape) == 3: s_d = s_d.squeeze(1)
        if len(s_s.shape) == 3: s_s = s_s.squeeze(1)
        action_logits = self.model.forward(s, s_d, s_s).squeeze(1)
        output[SampleBatch.VF_PREDS] = self.model._value_out.view(-1)
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        return output


class CILv2_RLModule_PT_Policy(TorchRLModule, PPORLModule):
    ''' RLModule that calculated the pretrained action dist at exploration '''
    def setup(self):
        self.model = CIL_multiview_actor_critic_RLModule(self.config.model_config_dict)

        # fixed pretrained model
        self.pt_model = [CIL_multiview_actor_critic_RLModule(
            self.config.model_config_dict).requires_grad_(False)]
        self.pt_model_device_set = False

        # for now we use the model config for passing the distribution, the
        # proper way is the catalog class
        dist = self.config.model_config_dict.get('output_distribution', 'gaussian') 
        if dist == 'beta':
            self.action_dist_cls = TorchBetaDistribution
        elif dist == 'gaussian':
            self.action_dist_cls = TorchDiagGaussian
        else:
            raise ValueError(
                f'{dist} distribution not supported. Use "beta" or "gaussian"'
            )

    def get_initial_state(self) -> dict:
        return {}

    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}
        s, s_d, s_s = restore_original_dimensions(
            batch[SampleBatch.OBS],
            self.config.observation_space,
            tensorlib=torch
        )
        if len(s_d.shape) == 3: s_d = s_d.squeeze(1)
        if len(s_s.shape) == 3: s_s = s_s.squeeze(1)
        with torch.no_grad():
            action_logits = self.model.forward(s, s_d, s_s).squeeze(1)
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        return output

    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        output = {}
        s, s_d, s_s = restore_original_dimensions(
            batch[SampleBatch.OBS],
            self.config.observation_space,
            tensorlib=torch
        )
        if len(s_d.shape) == 3: s_d = s_d.squeeze(1)
        if len(s_s.shape) == 3: s_s = s_s.squeeze(1)
        with torch.no_grad():
            action_logits = self.model.forward(s, s_d, s_s).squeeze(1)
        output[SampleBatch.VF_PREDS] = self.model._value_out.view(-1)
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        return output

    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        if not self.pt_model_device_set:
            self.pt_model[0].to(next(self.model.parameters()).device)
            self.pt_model_device_set = True
        output = {}
        s, s_d, s_s = restore_original_dimensions(
            batch[SampleBatch.OBS],
            self.config.observation_space,
            tensorlib=torch
        )
        if len(s_d.shape) == 3: s_d = s_d.squeeze(1)
        if len(s_s.shape) == 3: s_s = s_s.squeeze(1)
        action_logits = self.model.forward(s, s_d, s_s).squeeze(1)
        output[SampleBatch.VF_PREDS] = self.model._value_out.view(-1)
        output[SampleBatch.ACTION_DIST_INPUTS] = action_logits
        with torch.no_grad():
            pt_action_logits = self.pt_model[0].forward(s, s_d, s_s).squeeze(1)
        output['pretrained_action_dist'] = pt_action_logits
        return output

