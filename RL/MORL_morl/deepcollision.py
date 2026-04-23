import json
import os
import math
import random
import time
from collections import namedtuple, deque
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from RL.MORL_morl.networks import NatureCNN, mlp, layer_init, polyak_update
from RL.MORL_morl.morl_algorithm import MOAgent
from RL.MORL_morl.utils import linearly_decaying_value

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNet(nn.Module):
    """Single-objective Q-Network."""

    def __init__(self, obs_shape, obs_dim, action_dim, net_arch, drop_rate):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            obs_dim: number of observations
            action_dim: number of actions
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        if len(obs_shape) == 1:
            self.feature_extractor = None
            input_dim = obs_dim
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
            input_dim = self.feature_extractor.features_dim
        # |S| -> ... -> |A|
        self.net = mlp(input_dim, action_dim, net_arch, drop_rate=drop_rate)
        self.apply(layer_init)

    def forward(self, obs):
        """Predict Q values for all actions.

        Args:
            obs: current observation

        Returns: the Q values for all actions
        """
        if self.feature_extractor is not None:
            input = self.feature_extractor(obs)
        else:
            input = obs
        q_values = self.net(input)
        return q_values


class DeepCollision(MOAgent):

    def __init__(
            self,
            n_states: int = None,
            n_actions: int = None,
            n_rewards: int = 1,
            single_objective: List = ["distance", "time_to_collision"],
            scenarios: str = '',
            eval: bool = False,
            evaluations: int = None,
            learning_rate: float = 1e-2,
            initial_epsilon: float = 0.01,
            final_epsilon: float = 0.01,
            epsilon_decay_steps: int = None,  # None == fixed epsilon
            tau: float = 1.0,
            target_net_update_freq: int = 200,  # ignored if tau != 1.0
            buffer_size: int = int(1e6),
            net_arch: List = [256, 256, 256, 256],
            drop_rate: float = 0.0,
            batch_size: int = 256,
            learning_starts: int = 100,
            gradient_updates: int = 1,
            gamma: float = 0.99,
            max_grad_norm: Optional[float] = 1.0,
            experiment_name: str = "DeepCollision",
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
    ):
        """DeepCollision algorithm.

        Args:
            learning_rate: The learning rate (alpha).
            initial_epsilon: The initial epsilon value for epsilon-greedy exploration.
            final_epsilon: The final epsilon value for epsilon-greedy exploration.
            epsilon_decay_steps: The number of steps to decay epsilon over.
            tau: The soft update coefficient (keep in [0, 1]).
            target_net_update_freq: The frequency with which the target network is updated.
            buffer_size: The size of the replay buffer.
            net_arch: The size of the hidden layers of the value net.
            batch_size: The size of the batch to sample from the replay buffer.
            learning_starts: The number of steps before learning starts i.e. the agent will be random until learning starts.
            gradient_updates: The number of gradient updates per step.
            gamma: The discount factor (gamma).
            max_grad_norm: The maximum norm for the gradient clipping. If None, no gradient clipping is applied.
            experiment_name: The name of the experiment.
            seed: The seed for the random number generator.
            device: The device to use for training.
        """

        MOAgent.__init__(self, n_states=n_states, n_actions=n_actions, n_rewards=n_rewards, device=device, seed=seed)

        self.single_objective = single_objective
        self.eval = eval
        self.evaluations = evaluations
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.final_epsilon = final_epsilon
        self.tau = tau
        self.target_net_update_freq = target_net_update_freq
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gradient_updates = gradient_updates
        self.experiment_name = experiment_name

        self.q_net = QNet(self.observation_shape, self.observation_dim, self.action_dim, net_arch=net_arch,
                          drop_rate=drop_rate).to(self.device)
        self.target_q_net = QNet(self.observation_shape, self.observation_dim, self.action_dim, net_arch=net_arch,
                                 drop_rate=drop_rate).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        self.q_optim = optim.AdamW(self.q_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.replay_buffer = ReplayMemory(buffer_size)

        objectives_str = "_".join(self.single_objective)
        self.log_file_n = f"log_{int(time.time())}_{self.experiment_name}_{scenarios}_{objectives_str}.csv"
        print("log_file", self.log_file_n)
        self.header = (
                ['episode', 'step', 'action'] +
                self.single_objective +
                ['collision', 'collision_type'] +
                [f'reward_{obj}' for obj in self.single_objective] +
                ['reward_total', 'loss', 'done', 'tick', 'system_time', 'game_time']
        )
        self.df = pd.DataFrame(columns=self.header)

        self.header_pareto = (
                ['episode'] +
                [f'pareto_compute_{obj}' for obj in self.single_objective]
        )
        self.df_pareto = pd.DataFrame(columns=self.header_pareto)

        self.all_episode_scenario = {}

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "initial_epsilon": self.initial_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "clip_grand_norm": self.max_grad_norm,
            "target_net_update_freq": self.target_net_update_freq,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def write_to_file(self, train_results):
        if len(train_results) == len(self.header):
            train_results_cpu = [value.cpu().numpy() if isinstance(value, torch.Tensor) else value for value in
                                 train_results]
            train_results_cpu = [
                value.item() if isinstance(value, np.ndarray) and value.size == 1 else value
                for value in train_results_cpu
            ]
            self.df = self.df.append(pd.Series(train_results_cpu, index=self.df.columns), ignore_index=True)
        else:
            print("the length of the train_results is not the same as the columns")
            print(train_results)

    def save_to_file(self, save_dir: str = "train_results/"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.df.to_csv(save_dir + "/" + self.log_file_n, index=False)

    def write_pareto_to_file(self, pareto_results):
        if len(pareto_results) == len(self.header_pareto):
            pareto_results_cpu = [value.cpu().numpy() if isinstance(value, torch.Tensor) else value for value in pareto_results]
            self.df_pareto = self.df_pareto.append(pd.Series(pareto_results_cpu, index=self.df_pareto.columns), ignore_index=True)
        else:
            print("the length of the pareto_results is not the same as the columns")
            print(pareto_results)

    def save_pareto_to_file(self, save_dir: str = "train_results/"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.df_pareto.to_csv(save_dir + "/" + self.log_file_n.split('.csv')[0] + "_pareto.csv", index=False)

    def update_all_episode_scenario(self, all_tick_scenario):
        self.all_episode_scenario.update(all_tick_scenario)

    def save_all_episode_scenario(self, save_dir: str = "train_results/"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + "/" + self.log_file_n.split('.csv')[0] + "_all_episode_scenario.json", 'w') as f:
            json.dump(self.all_episode_scenario, f, indent=4)

    def save(self, save_replay_buffer: bool = True, save_dir: str = "train_results/", filename: Optional[str] = None):
        """Save the model and the replay buffer if specified.

        Args:
            save_replay_buffer: Whether to save the replay buffer too.
            save_dir: Directory to save the model.
            filename: filename to save the model.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        saved_params = {}
        saved_params["q_net_state_dict"] = self.q_net.state_dict()

        saved_params["q_net_optimizer_state_dict"] = self.q_optim.state_dict()
        if save_replay_buffer:
            saved_params["replay_buffer"] = self.replay_buffer
        filename = self.log_file_n.split('.csv')[0] if filename is None else filename
        torch.save(saved_params, save_dir + "/" + filename + ".tar")

    def load(self, path: str, load_replay_buffer: bool = True):
        """Load the model and the replay buffer if specified.

        Args:
            path: Path to the model.
            load_replay_buffer: Whether to load the replay buffer too.
        """
        params = torch.load(path)
        self.q_net.load_state_dict(params["q_net_state_dict"])
        self.target_q_net.load_state_dict(params["q_net_state_dict"])
        self.q_optim.load_state_dict(params["q_net_optimizer_state_dict"])
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]

    def update(self):
        critic_losses = []
        for g in range(self.gradient_updates):
            transitions = self.replay_buffer.sample(self.batch_size)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            while all(s is None for s in batch.next_state):
                transitions = self.replay_buffer.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.q_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_q_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            critic_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            self.q_optim.zero_grad()
            critic_loss.backward()
            # In-place gradient clipping
            # if self.max_grad_norm is not None:
            #     torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)

        # if self.epsilon_decay_steps is not None:
        #     self.epsilon = linearly_decaying_value(
        #         self.initial_epsilon,
        #         self.epsilon_decay_steps,
        #         self.global_step,
        #         self.learning_starts,
        #         self.final_epsilon,
        #     )

        return critic_losses

    def calcu_eps(self):
        if self.eval:
            eps_threshold = self.final_epsilon
        else:
            eps_threshold = self.final_epsilon + (self.initial_epsilon - self.final_epsilon) * math.exp(
                -1. * self.global_step / self.epsilon_decay_steps)
        return eps_threshold

    def act(self, obs):
        """Epsilon-greedily select an action given an observation.

        Args:
            obs: observation

        Returns: an action to take.
        """
        sample = random.random()
        eps_threshold = self.calcu_eps()
        # if self.eval:
        #     eps_threshold = self.final_epsilon
        # else:
        #     eps_threshold = linearly_decaying_value(
        #             self.initial_epsilon,
        #             self.epsilon_decay_steps,
        #             self.global_step,
        #             self.learning_starts,
        #             self.final_epsilon,
        #         )
        if sample > eps_threshold:
            with torch.no_grad():
                return self.q_net(obs).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
