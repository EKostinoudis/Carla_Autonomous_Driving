import gymnasium as gym
from math import log
import torch
from torch.distributions.beta import Beta
import numpy as np

from ray.rllib.models.torch.torch_distributions import TorchDistribution, TorchDeterministic
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import TensorType

def stabilize(x):
    x = torch.clamp(x, log(SMALL_NUMBER), -log(SMALL_NUMBER))
    return torch.log(torch.exp(x) + 1.0) + 1.0

class TorchBetaDistribution(TorchDistribution):
    def __init__(self, alpha, beta, low: float = -1.0, high: float = 1.0):
        # Stabilize input parameters (possibly coming from a linear layer).
        self.alpha = stabilize(alpha)
        self.beta = stabilize(beta)
        self.low = low
        self.high = high
        super().__init__(self.alpha, self.beta)

    def _get_torch_distribution(self, alpha, beta) -> "torch.distributions.Distribution":
        return Beta(concentration1=alpha, concentration0=beta)

    def sample(self, *, sample_shape=torch.Size()):
        sample = self._dist.sample(sample_shape)
        return self._squash(sample)

    def rsample(self, *, sample_shape=torch.Size()):
        rsample = self._dist.rsample(sample_shape)
        return self._squash(rsample)

    def logp(self, value: TensorType) -> TensorType:
        return super().logp(self._unsquash(value)).sum(-1)

    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

    def kl(self, other: "TorchDistribution") -> TensorType:
        return super().kl(other).sum(-1)

    @staticmethod
    def required_input_dim(space: gym.Space, **kwargs) -> int:
        assert isinstance(space, gym.spaces.Box)
        return int(np.prod(space.shape, dtype=np.int32) * 2)

    @classmethod
    def from_logits(cls, logits: TensorType, **kwargs) -> "TorchBetaDistribution":
        alpha, beta = logits.chunk(2, dim=-1)
        return TorchBetaDistribution(alpha=alpha, beta=beta, **kwargs)

    def to_deterministic(self) -> "TorchDeterministic":
        mean = self.alpha / (self.alpha + self.beta)
        return TorchDeterministic(self._squash(mean))

    def _squash(self, raw_values: TensorType) -> TensorType:
        return raw_values * (self.high - self.low) + self.low

    def _unsquash(self, values: TensorType) -> TensorType:
        return (values - self.low) / (self.high - self.low)

