from distutils.dist import Distribution

import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy, MultiInputActorCriticPolicy
from typing import Tuple, Callable, Any
from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor
from typing import List, Optional, Type, Union, Dict

from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
# from torchvision import models


class CustomMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):
    recurrent_alias: Dict = {"GRU": th.nn.GRU}
    """
     MultiInputActorClass policy class for actor-critic algorithms (has both policy and value prediction).
     Used by A2C, PPO and the likes.

     :param observation_space: Observation space (Tuple)
     :param action_space: Action space
     :param lr_schedule: Learning rate schedule (could be constant)
     :param net_arch: The specification of the policy and value networks.
     :param activation_fn: Activation function
     :param ortho_init: Whether to use or not orthogonal initialization
     :param use_sde: Whether to use State Dependent Exploration or not
     :param log_std_init: Initial value for the log standard deviation
     :param full_std: Whether to use (n_features x n_actions) parameters
         for the std instead of only (n_features,) when using gSDE
     :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
         a positive standard deviation (cf paper). It allows to keep variance
         above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
     :param squash_output: Whether to squash the output using a tanh function,
         this allows to ensure boundaries when using gSDE.
     :param features_extractor_class: Uses the CombinedExtractor
     :param features_extractor_kwargs: Keyword arguments
         to pass to the features extractor.
     :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
     :param normalize_images: Whether to normalize images or not,
          dividing by 255.0 (True by default)
     :param optimizer_class: The optimizer to use,
         ``th.optim.Adam`` by default
     :param optimizer_kwargs: Additional keyword arguments,
         excluding the learning rate, to pass to the optimizer
     """
    
        
    def __init__(
            self,
            observation_space: spaces.Dict,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,

            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            features_extractor_class = None,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            pi_features_extractor_class: Type[BaseFeaturesExtractor] = None,
            pi_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            vf_features_extractor_class: Type[BaseFeaturesExtractor] = None,
            vf_features_extractor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if features_extractor_class is None:
            features_extractor_class = pi_features_extractor_class
            features_extractor_kwargs = pi_features_extractor_kwargs
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        if hasattr(self.features_extractor, "recurrent_extractor"):
            self.mlp_extractor.policy_net[0] = nn.Linear(self.features_extractor.recurrent_extractor.hidden_size,
                                                         self.mlp_extractor.policy_net[0].out_features)
            self.mlp_extractor.value_net[0] = nn.Linear(self.features_extractor.recurrent_extractor.hidden_size,
                                                        self.mlp_extractor.value_net[0].out_features)
        if features_extractor_class is not None:
            pass
        else:
            assert pi_features_extractor_class is not None and vf_features_extractor_class is not None
            self.pi_features_extractor = self.make_features_extractor(pi_features_extractor_class, pi_features_extractor_kwargs)
            self.vf_features_extractor = self.make_features_extractor(vf_features_extractor_class, vf_features_extractor_kwargs)
            
            test = 1
        self._build(lr_schedule)
            
    def forward(self, obs: th.Tensor, deterministic: bool = False, latent: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :param latent: latent feature
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        if hasattr(self.features_extractor, "recurrent_extractor"):
            features, h = self.extract_features(obs)
        else:
            features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        if hasattr(self.features_extractor, "recurrent_extractor"):
            return actions, values, log_prob, h
        else:
            return actions, values, log_prob

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        if hasattr(self.features_extractor, "recurrent_extractor"):
            features, h = self.extract_features(obs)
        else:
            features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs, latent: th.Tensor = None) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        if hasattr(self.features_extractor, "recurrent_extractor"):
            features, h = super().extract_features(obs, self.vf_features_extractor)
            latent_pi = self.mlp_extractor.forward_actor(features)
            return self._get_action_dist_from_latent(latent_pi)
        else:
            features = super().extract_features(obs, self.vf_features_extractor)
            latent_pi = self.mlp_extractor.forward_actor(features)
            return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        if hasattr(self.features_extractor, "recurrent_extractor"):
            features = super().extract_features(obs, self.vf_features_extractor)[0]
        else:
            features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    # def _predict(self, observation, deterministic = False):
    #     if hasattr(self.features_extractor, "recurrent_extractor"):
    #         dist, h = self.get_distribution(observation)
    #         return dist.get_actions(deterministic=deterministic), h
    #     else:
    #         dist = self.get_distribution(observation)
    #         return dist.get_actions(deterministic=deterministic)


def debug():
    test = 1


if __name__ == "__main__":
    debug()
