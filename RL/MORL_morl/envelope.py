import json
import os
import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F

from RL.MORL_morl.prioritized_buffer import PrioritizedReplayBuffer
from RL.MORL_morl.networks import NatureCNN, mlp, layer_init, polyak_update
from RL.MORL_morl.morl_algorithm import MOAgent
from RL.MORL_morl.replay_buffer import ReplayBuffer
from RL.MORL_morl.utils import linearly_decaying_value
from RL.MORL_morl.weights import random_weights


class QNet(nn.Module):
    """Multi-objective Q-Network conditioned on the weight vector."""

    def __init__(self, obs_shape, obs_dim, action_dim, rew_dim, net_arch, drop_rate):
        """Initialize the Q network.

        Args:
            obs_shape: shape of the observation
            obs_dim: number of observations
            action_dim: number of actions
            rew_dim: number of objectives
            net_arch: network architecture (number of units per layer)
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rew_dim = rew_dim

        if len(obs_shape) == 1:
            self.feature_extractor = None
            input_dim = obs_dim + rew_dim
        elif len(obs_shape) > 1:  # Image observation
            self.feature_extractor = NatureCNN(self.obs_shape, features_dim=512)
            input_dim = self.feature_extractor.features_dim + rew_dim
        # |S| + |R| -> ... -> |A| * |R|
        self.net = mlp(input_dim, action_dim * rew_dim, net_arch, drop_rate=drop_rate)
        self.apply(layer_init)

    def forward(self, obs, w):
        """Predict Q values for all actions.

        Args:
            obs: current observation
            w: weight vector

        Returns: the Q values for all actions
        """
        if self.feature_extractor is not None:
            features = self.feature_extractor(obs)
            if w.dim() == 1:
                w = w.unsqueeze(0)
            input = torch.cat((features, w), dim=features.dim() - 1)
        else:
            input = torch.cat((obs, w), dim=w.dim() - 1)
        q_values = self.net(input)
        return q_values.view(-1, self.action_dim, self.rew_dim)  # Batch size X Actions X Rewards


class Envelope(MOAgent):

    def __init__(
            self,
            n_states: int = None,
            n_actions: int = None,
            n_rewards: int = None,
            multi_objective: List = ["distance", "time_to_collision"],  # distance  time_to_collision  comfort  completion
            scenarios: str = '',
            eval: bool = False,
            evaluations: int = None,
            learning_rate: float = 3e-4,
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
            envelope: bool = True,
            num_sample_w: int = 4,
            per: bool = True,
            per_alpha: float = 0.6,
            initial_homotopy_lambda: float = 0.0,
            final_homotopy_lambda: float = 1.0,
            homotopy_decay_steps: int = None,
            experiment_name: str = "Envelope",
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
    ):
        """Envelope Q-learning algorithm.

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
            envelope: Whether to use the envelope method.
            num_sample_w: The number of weight vectors to sample for the envelope target.
            per: Whether to use prioritized experience replay.
            per_alpha: The alpha parameter for prioritized experience replay.
            initial_homotopy_lambda: The initial value of the homotopy parameter for homotopy optimization.
            final_homotopy_lambda: The final value of the homotopy parameter.
            homotopy_decay_steps: The number of steps to decay the homotopy parameter over.
            experiment_name: The name of the experiment.
            seed: The seed for the random number generator.
            device: The device to use for training.
        """

        MOAgent.__init__(self, n_states=n_states, n_actions=n_actions, n_rewards=n_rewards, device=device, seed=seed)

        self.multi_objective = multi_objective
        self.evaluations = evaluations
        self.learning_rate = learning_rate
        self.initial_epsilon = initial_epsilon
        if eval:
            self.epsilon = final_epsilon
        else:
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
        self.per = per
        self.per_alpha = per_alpha
        self.gradient_updates = gradient_updates
        self.initial_homotopy_lambda = initial_homotopy_lambda
        self.final_homotopy_lambda = final_homotopy_lambda
        self.homotopy_decay_steps = homotopy_decay_steps
        self.experiment_name = experiment_name

        self.q_net = QNet(self.observation_shape, self.observation_dim, self.action_dim, self.reward_dim,
                          net_arch=net_arch, drop_rate=drop_rate).to(self.device)
        self.target_q_net = QNet(self.observation_shape, self.observation_dim, self.action_dim, self.reward_dim,
                                 net_arch=net_arch, drop_rate=drop_rate).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        for param in self.target_q_net.parameters():
            param.requires_grad = False

        self.q_optim = optim.AdamW(self.q_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.envelope = envelope
        self.num_sample_w = num_sample_w
        self.homotopy_lambda = self.initial_homotopy_lambda
        if self.per:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.observation_shape,
                self.observation_dim,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                # action_dtype=np.uint8,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.observation_shape,
                self.observation_dim,
                1,
                rew_dim=self.reward_dim,
                max_size=buffer_size,
                # action_dtype=np.uint8,
            )

        objectives_str = "_".join(self.multi_objective)
        self.log_file_n = f"log_{int(time.time())}_{self.experiment_name}_{scenarios}_{objectives_str}.csv"
        print("log_file", self.log_file_n)
        self.header = (
                ['episode', 'step', 'action'] +
                self.multi_objective +
                ['collision', 'collision_type'] +
                [f'reward_{obj}' for obj in self.multi_objective] +
                ['reward_total', 'loss', 'done', 'tick', 'system_time', 'game_time']
        )
        self.df = pd.DataFrame(columns=self.header)

        self.header_pareto = (
                ['episode'] +
                [f'pareto_compute_{obj}' for obj in self.multi_objective] +
                [f'pareto_predict_{obj}' for obj in self.multi_objective] +
                [f'pareto_predict_0_{obj}' for obj in self.multi_objective]
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
            "use_envelope": self.envelope,
            "num_sample_w": self.num_sample_w,
            "net_arch": self.net_arch,
            "per": self.per,
            "gradient_updates": self.gradient_updates,
            "buffer_size": self.buffer_size,
            "initial_homotopy_lambda": self.initial_homotopy_lambda,
            "final_homotopy_lambda": self.final_homotopy_lambda,
            "homotopy_decay_steps": self.homotopy_decay_steps,
            "learning_starts": self.learning_starts,
            "seed": self.seed,
        }

    def write_to_file(self, train_results):
        if len(train_results) == len(self.header):
            train_results_cpu = [value.cpu().numpy() if isinstance(value, torch.Tensor) else value for value in train_results]
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

    def _sample_batch_experiences(self):
        return self.replay_buffer.sample(self.batch_size, to_tensor=True, device=self.device)

    def update(self):
        critic_losses = []
        for g in range(self.gradient_updates):
            if self.per:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                    b_inds,
                ) = self._sample_batch_experiences()
            else:
                (
                    b_obs,
                    b_actions,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                ) = self._sample_batch_experiences()

            sampled_w = (
                torch.tensor(random_weights(dim=self.reward_dim, n=self.num_sample_w, dist="gaussian", rng=self.np_random))
                .float()
                .to(self.device)
            )  # sample num_sample_w random weights
            w = sampled_w.repeat_interleave(b_obs.size(0), 0)  # repeat the weights for each sample
            if len(self.observation_shape) == 1:
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = (
                    b_obs.repeat(self.num_sample_w, 1),
                    b_actions.repeat(self.num_sample_w, 1),
                    b_rewards.repeat(self.num_sample_w, 1),
                    b_next_obs.repeat(self.num_sample_w, 1),
                    b_dones.repeat(self.num_sample_w, 1),
                )
            elif len(self.observation_shape) > 1:  # Image observation
                b_obs, b_actions, b_rewards, b_next_obs, b_dones = (
                    b_obs.repeat(self.num_sample_w, *(1 for _ in range(b_obs.dim() - 1))),
                    b_actions.repeat(self.num_sample_w, 1),
                    b_rewards.repeat(self.num_sample_w, 1),
                    b_next_obs.repeat(self.num_sample_w, *(1 for _ in range(b_next_obs.dim() - 1))),
                    b_dones.repeat(self.num_sample_w, 1),
                )

            with torch.no_grad():
                if self.envelope:
                    target = self._envelope_target(b_next_obs, w, sampled_w)
                else:
                    target = self._ddqn_target(b_next_obs, w)
                target_q = b_rewards + (1 - b_dones) * self.gamma * target

            q_values = self.q_net(b_obs, w)
            q_value = q_values.gather(
                1,
                b_actions.long().reshape(-1, 1, 1).expand(q_values.size(0), 1, q_values.size(2)),
            )
            q_value = q_value.reshape(-1, self.reward_dim)

            critic_loss = F.smooth_l1_loss(q_value, target_q)

            if self.homotopy_lambda > 0:
                wQ = torch.einsum("br,br->b", q_value, w)
                wTQ = torch.einsum("br,br->b", target_q, w)
                auxiliary_loss = F.smooth_l1_loss(wQ, wTQ)
                critic_loss = (1 - self.homotopy_lambda) * critic_loss + self.homotopy_lambda * auxiliary_loss

            self.q_optim.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.q_optim.step()
            critic_losses.append(critic_loss.item())

            if self.per:
                td_err = (q_value[: len(b_inds)] - target_q[: len(b_inds)]).detach()
                priority = torch.einsum("sr,sr->s", td_err, w[: len(b_inds)]).abs()
                priority = priority.cpu().numpy().flatten()
                priority = (priority + self.replay_buffer.min_priority) ** self.per_alpha
                self.replay_buffer.update_priorities(b_inds, priority)

        if self.tau != 1 or self.global_step % self.target_net_update_freq == 0:
            polyak_update(self.q_net.parameters(), self.target_q_net.parameters(), self.tau)

        if self.epsilon_decay_steps is not None:
            self.epsilon = linearly_decaying_value(
                self.initial_epsilon,
                self.epsilon_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_epsilon,
            )

        if self.homotopy_decay_steps is not None:
            self.homotopy_lambda = linearly_decaying_value(
                self.initial_homotopy_lambda,
                self.homotopy_decay_steps,
                self.global_step,
                self.learning_starts,
                self.final_homotopy_lambda,
            )

        return critic_losses

    @torch.no_grad()
    def _envelope_target(self, obs: torch.Tensor, w: torch.Tensor, sampled_w: torch.Tensor) -> torch.Tensor:
        """Computes the envelope target for the given observation and weight.

        Args:
            obs: current observation.
            w: current weight vector.
            sampled_w: set of sampled weight vectors (>1!).

        Returns: the envelope target.
        """
        # Repeat the weights for each sample
        W = sampled_w.repeat(obs.size(0), 1)
        # Repeat the observations for each sampled weight
        next_obs = obs.repeat_interleave(sampled_w.size(0), 0)
        # Batch size X Num sampled weights X Num actions X Num objectives
        next_q_values = self.q_net(next_obs, W).view(obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim)
        # Scalarized Q values for each sampled weight
        scalarized_next_q_values = torch.einsum("br,bwar->bwa", w, next_q_values)
        # Max Q values for each sampled weight
        max_q, ac = torch.max(scalarized_next_q_values, dim=2)
        # Max weights in the envelope
        pref = torch.argmax(max_q, dim=1)

        # MO Q-values evaluated on the target network
        next_q_values_target = self.target_q_net(next_obs, W).view(
            obs.size(0), sampled_w.size(0), self.action_dim, self.reward_dim
        )

        # Index the Q-values for the max actions
        max_next_q = next_q_values_target.gather(
            2,
            ac.unsqueeze(2).unsqueeze(3).expand(next_q_values.size(0), next_q_values.size(1), 1, next_q_values.size(3)),
        ).squeeze(2)
        # Index the Q-values for the max sampled weights
        max_next_q = max_next_q.gather(1, pref.reshape(-1, 1, 1).expand(max_next_q.size(0), 1,
                                                                        max_next_q.size(2))).squeeze(1)
        return max_next_q

    @torch.no_grad()
    def _ddqn_target(self, obs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Double DQN target for the given observation and weight.

        Args:
            obs: observation
            w: weight vector.

        Returns: the DQN target.
        """
        # Max action for each state
        q_values = self.q_net(obs, w)
        scalarized_q_values = torch.einsum("br,bar->ba", w, q_values)
        max_acts = torch.argmax(scalarized_q_values, dim=1)
        # Action evaluated with the target network
        q_values_target = self.target_q_net(obs, w)
        q_values_target = q_values_target.gather(
            1,
            max_acts.long().reshape(-1, 1, 1).expand(q_values_target.size(0), 1, q_values_target.size(2)),
        )
        q_values_target = q_values_target.reshape(-1, self.reward_dim)
        return q_values_target

    def eval(self, obs: np.ndarray, w: np.ndarray) -> int:
        obs = torch.as_tensor(obs).float().to(self.device)
        w = torch.as_tensor(w).float().to(self.device)
        return self.max_action(obs, w)

    def act(self, obs: torch.Tensor, w: torch.Tensor) -> int:
        """Epsilon-greedily select an action given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: an integer representing the action to take.
        """
        if self.np_random.random() < self.epsilon:
            return int(np.random.choice(self.action_dim, 1)[0])
        else:
            return self.max_action(obs, w)

    @torch.no_grad()
    def max_action(self, obs: torch.Tensor, w: torch.Tensor) -> int:
        """Select the action with the highest Q-value given an observation and weight.

        Args:
            obs: observation
            w: weight vector

        Returns: the action with the highest Q-value.
        """
        q_values = self.q_net(obs, w)
        scalarized_q_values = torch.einsum("r,bar->ba", w, q_values)
        max_act = torch.argmax(scalarized_q_values, dim=1)
        return max_act.detach().item()

    def predict(self, state, probe, w_num=1):
        if state is None:
            obs = torch.zeros((w_num, self.observation_dim), dtype=torch.float32).to(self.device)
        else:
            obs = torch.as_tensor(state).float().to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        w = torch.as_tensor(probe).float().to(self.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        q_values = self.q_net(obs, w)

        Q = q_values.detach().view(-1, self.reward_dim)
        total_pairs = w.size(0)
        assert total_pairs % w_num == 0, "w size[0] must be divisible by w_num"
        s_num = total_pairs // w_num
        hq = self.H(Q, w, s_num, w_num)

        return hq

    def H(self, Q, w, s_num, w_num):
        # 1. mask for reordering the batch
        mask = torch.cat([
            torch.arange(i, s_num * w_num + i, s_num)
            for i in range(s_num)
        ]).long().to(self.device)

        reQ = Q.view(-1, self.action_dim * self.reward_dim)[mask]
        reQ = reQ.view(-1, self.reward_dim)

        # 2. extend Q and preference weight batches
        reQ_ext = reQ.repeat(w_num, 1)  # (s_num * w_num * action_size, reward_size)

        w_ext = w.unsqueeze(2).repeat(1, self.action_dim * w_num, 1)
        w_ext = w_ext.view(-1, self.reward_dim)

        # 3. compute inner product: Q · w
        prod = torch.bmm(reQ_ext.unsqueeze(1), w_ext.unsqueeze(2)).squeeze(1).squeeze(1)

        # 4. reshape to group by state: weight combos, then take max
        prod = prod.view(-1, self.action_dim * w_num)
        inds = prod.argmax(dim=1, keepdim=True)

        # 5. build mask using one-hot indexing
        mask = torch.zeros_like(prod, dtype=torch.bool)
        mask.scatter_(1, inds, True)
        mask = mask.view(-1, 1).repeat(1, self.reward_dim)

        # 6. select best Q-values according to mask
        HQ = reQ_ext[mask].view(-1, self.reward_dim)

        return HQ
