from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch


class MOAgent(ABC):
    """An MORL Agent, can contain one or multiple MOPolicies. Contains helpers to extract features from the environment."""

    def __init__(self, n_states, n_actions, n_rewards, device: Union[torch.device, str] = "auto", seed: Optional[int] = None) -> None:
        """Initializes the agent.

        Args:
            n_states: (int): The number of states/observations.
            n_rewards: (int): The number of rewards/objectives.
            device: (str): The device to use for training. Can be "auto", "cpu" or "cuda".
            seed: (int): The seed to use for the random number generator
        """
        # Continuous parameter observation space
        self.observation_shape = (1,)
        self.observation_dim = n_states
        # Discrete action space
        self.action_dim = n_actions
        self.action_shape = (1,)
        # Reward
        self.reward_dim = n_rewards

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

        self.global_step = 0
        self.num_episodes = 0
        self.seed = seed
        self.np_random = np.random.default_rng(self.seed)

    def generate_action_space(self):
        v_horizontal = [-20, 0, 20]
        # v_horizontal = [-20, -10, 0, 10, 20]
        v_vertical = [-3.5, 0, 3.5]
        v_behavior = [0, 0.1, 0.2]

        p_horizontal = [10]
        # p_horizontal = [10, 15]
        p_vertical = [-10, 10]
        # p_direction_x = [-1, 0, 1]
        p_direction_y = [-1, 0, 1]
        p_speed = [0.94, 1.43]

        action_space = []
        self.v_actions = 0
        for h in v_horizontal:
            for v in v_vertical:
                if h == v == 0:
                    continue
                for b in v_behavior:
                    action_space.append([h, v, b])
                    self.v_actions += 1
        self.p_actions = 0
        for h in p_horizontal:
            for v in p_vertical:
                if h == v == 0:
                    continue
                for s in p_speed:
                    if v > 0:
                        x = 1
                        for y in p_direction_y:
                            action_space.append([h, v, x, y, s])
                            self.p_actions += 1
                    else:
                        x = -1
                        for y in p_direction_y:
                            action_space.append([h, v, x, y, s])
                            self.p_actions += 1

        print("v_actions", self.v_actions, "p_actions", self.p_actions, "total_actions", self.v_actions + self.p_actions)
        return action_space, self.v_actions + self.p_actions
        # return action_space, self.v_actions
        # print("p_actions", self.p_actions)
        # return action_space, self.p_actions

