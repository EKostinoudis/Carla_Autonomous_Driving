from typing import Mapping, Any
import torch

from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule

from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.models.modelv2 import restore_original_dimensions

from .beta_distribution import TorchBetaDistribution
from .network.models.architectures.CIL_multiview.CIL_multiview_rllib import CIL_multiview_actor_critic_RLModule

class CILv2_RLModule(TorchRLModule, PPORLModule):
    def setup(self):
        self.model = CIL_multiview_actor_critic_RLModule(self.config.model_config_dict)
        self.action_dist_cls = TorchBetaDistribution

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


