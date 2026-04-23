import json
import os
import time
from typing import Optional, Union, List

import pandas as pd
import torch

from RL.MORL_morl.morl_algorithm import MOAgent


class RandomAlgorithm(MOAgent):

    def __init__(
            self,
            n_states: int = None,
            n_actions: int = None,
            n_rewards: int = None,
            objective: List = ["distance", "time_to_collision"],
            scenarios: str = '',
            evaluations: int = None,
            gamma: float = 0.99,
            experiment_name: str = "Random",
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
    ):
        """Random algorithm.

        Args:
            experiment_name: The name of the experiment.
            seed: The seed for the random number generator.
            device: The device to use for training.
        """

        MOAgent.__init__(self, n_states=n_states, n_actions=n_actions, n_rewards=n_rewards, device=device, seed=seed)

        self.objective = objective
        self.evaluations = evaluations
        self.gamma = gamma
        self.experiment_name = experiment_name

        objectives_str = "_".join(self.objective)
        self.log_file_n = f"log_{int(time.time())}_{self.experiment_name}_{scenarios}_{objectives_str}.csv"
        print("log_file", self.log_file_n)
        self.header = (
                ['episode', 'step', 'action'] +
                self.objective +
                ['collision', 'collision_type'] +
                [f'reward_{obj}' for obj in self.objective] +
                ['reward_total', 'loss', 'done', 'tick', 'system_time', 'game_time']
        )
        self.df = pd.DataFrame(columns=self.header)

        self.header_pareto = (
                ['episode'] +
                [f'pareto_compute_{obj}' for obj in self.objective]
        )
        self.df_pareto = pd.DataFrame(columns=self.header_pareto)

        self.all_episode_scenario = {}

    def get_config(self):
        return {
            "gamma": self.gamma,
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
        pass
