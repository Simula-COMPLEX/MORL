import argparse
import json
import os
import optuna
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime

import torch

import gc

from optuna.trial import TrialState

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

sys.path.insert(0, '/MORL/scenario_runner')
sys.path.insert(0, '/MORL/leaderboard')
sys.path.insert(0, '/MORL/carla/PythonAPI/carla')
sys.path.insert(0, '/MORL')

from RL.MORL_morl.envelope import Envelope
from RL.MORL_morl.deepcollision import DeepCollision
from RL.MORL_morl.random_algorithm import RandomAlgorithm

from leaderboard.leaderboard_evaluator_morl import LeaderboardEvaluator

from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.statistics_manager import StatisticsManager

PATH = '/MORL'


class RLGeneration:

    def __init__(self, evaluator, args):
        """
        Problem Initialization.

        :param int n_variables: the problem size. The number of float numbers in the solution vector (list).
        """

        n_states = 10 + 2 * 15
        n_actions = 12
        tau = 1.0
        target_net_update_freq = 30  # 1000,  # 500 reduce by gradient updates
        gradient_updates = 1
        gamma = 0.99

        objectives = args.objective.split('+')  # distance+time_to_collision
        # multi_objective = [objectives[0], objectives[-1]]
        # multi_objective = [objective for objective in objectives]
        n_rewards = len(objectives)

        if args.algorithm == 'envelope':
            self.agent = Envelope(
                n_states=n_states,
                n_actions=n_actions,
                n_rewards=n_rewards,
                multi_objective=objectives,
                scenarios=args.scenarios.split('/')[-1].split('.')[0],
                eval=args.eval,
                evaluations=args.evaluations,
                learning_rate=args.lr,
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay_steps=args.decay_steps,
                tau=tau,
                target_net_update_freq=target_net_update_freq,  # 1000,  # 500 reduce by gradient updates
                buffer_size=args.memory,
                net_arch=[(n_states+n_rewards) * 16, (n_states+n_rewards) * 32, (n_states+n_rewards) * 64, (n_states+n_rewards) * 32],
                drop_rate=args.drop_rate,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                gradient_updates=gradient_updates,
                gamma=gamma,
                max_grad_norm=args.max_grad_norm,
                envelope=True,
                num_sample_w=args.num_sample_w,
                initial_homotopy_lambda=0.0,
                final_homotopy_lambda=1.0,
                homotopy_decay_steps=args.decay_steps,
            )
        elif args.algorithm == 'deepcollision':
            self.agent = DeepCollision(
                n_states=n_states,
                n_actions=n_actions,
                single_objective=objectives,
                scenarios=args.scenarios.split('/')[-1].split('.')[0],
                eval=args.eval,
                evaluations=args.evaluations,
                learning_rate=args.lr,
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay_steps=args.decay_steps,
                tau=tau,
                target_net_update_freq=target_net_update_freq,  # 1000,  # 500 reduce by gradient updates
                buffer_size=args.memory,
                net_arch=[64, 128, 256, 128],
                drop_rate=args.drop_rate,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                gradient_updates=gradient_updates,
                gamma=gamma,
                max_grad_norm=1.0,
            )
        elif args.algorithm == 'random' or args.algorithm == 'action_replay':
            self.agent = RandomAlgorithm(
                n_states=n_states,
                n_actions=n_actions,
                objective=objectives,
                scenarios=args.scenarios.split('/')[-1].split('.')[0],
                evaluations=args.evaluations,
                gamma=gamma,
            )
        print("Configuration==============================================================")
        print(self.agent.get_config())
        print("Configuration==============================================================")

        self.evaluator = evaluator
        self.args = args
        self.set_route()

    def set_route(self):
        self.route_indexer = RouteIndexer(self.args.routes, self.args.scenarios, self.args.repetitions)

        if self.args.resume:
            self.route_indexer.resume(self.args.checkpoint)
            self.evaluator.statistics_manager.resume(self.args.checkpoint)
        else:
            self.evaluator.statistics_manager.clear_record(self.args.checkpoint)
            self.route_indexer.save_state(self.args.checkpoint)

        self.config = self.route_indexer.next()

    def train(self):
        for i in range(self.args.evaluations):
            self.evaluator._load_and_run_scenario(self.args, self.config, self.agent, episode=i)
            if i % 9 == 0:
                self.agent.save_to_file()
            if i == 0 or i == 999 or i == 999 + 50 or i == 999 + 100 or i == 999 + 150 or i == 999 + 200:
                self.agent.save(filename=f"{self.agent.log_file_n.split('.csv')[0]}_episode_{i+1}")
        self.agent.save_to_file()
        self.agent.save(filename=f"{self.agent.log_file_n.split('.csv')[0]}")
        gc.collect()

    def eval(self):
        if self.args.algorithm == 'envelope' or self.args.algorithm == 'deepcollision':
            all_files = os.listdir("eval_results/")
            for file in all_files:
                if (
                        file.endswith(".tar")
                        and self.args.scenario_id in file
                        and file.split('_')[2].lower() == self.args.algorithm
                        and all(obj in file for obj in self.args.objective.split('+'))
                ):
                    self.agent.load(path=f"eval_results/{file}")
                    print(f"load {self.args.algorithm} model: eval_results/{file}")
                    break
        for i in range(self.args.evaluations):
            self.evaluator._load_and_run_scenario(self.args, self.config, self.agent, episode=i)
            if i % 9 == 0:
                self.agent.save_to_file(save_dir="eval_results/")
                self.agent.save_pareto_to_file(save_dir="eval_results/")
        self.agent.save_to_file(save_dir="eval_results/")
        self.agent.save_pareto_to_file(save_dir="eval_results/")
        self.agent.save_all_episode_scenario(save_dir="eval_results/")
        gc.collect()


def main(args):
    """
    start
    """
    statistics_manager = StatisticsManager()

    L = 10

    try:
        leaderboard_evaluator = LeaderboardEvaluator(args, statistics_manager)

        rl_alg = RLGeneration(evaluator=leaderboard_evaluator, args=args)
        if args.eval:
            rl_alg.eval()
        else:
            rl_alg.train()
        # rl_alg.train()

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


def str_to_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False


def int_none(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for decay_steps: {value}")


if __name__ == '__main__':
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', required=False, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', required=False, default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='1',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--carlaProviderSeed', default='2000',
                        help='Seed used by the CarlaProvider (default: 2000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="120.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        default='{}/leaderboard/data/test_routes/scenario_4.xml'.format(PATH),
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        # required=True
                        )
    parser.add_argument('--scenarios',
                        default='{}/leaderboard/data/test_routes/scenario_4.json'.format(PATH),
                        help='Name of the scenario annotation file to be mixed with the route.',
                        # required=True
                        )
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent",
                        default='{}/leaderboard/team_code/interfuser_agent.py'.format(PATH),
                        type=str, help="Path to Agent's py file to evaluate",
                        # required=True
                        )
    parser.add_argument("--agent-config",
                        default='{}/leaderboard/team_code/interfuser_config.py'.format(PATH),
                        type=str, help="Path to Agent's configuration file")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        # '{}/simulation_results_{}.json'.format(args.checkpoint, str(int(time.time())))
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--checkpoint_path", type=str,
                        default=f'{PATH}/leaderboard/records/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_simulation_results.json',
                        # '{}/simulation_results_{}.json'.format(args.checkpoint, str(int(time.time())))
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--uq_path", type=str,
                        default=f'{PATH}/leaderboard/records/uncertainty_quantification/',
                        # default='{}/epiga/alg/results/uncertainty_quantification/'.format(PATH),
                        # '{}/simulation_results_{}.json'.format(args.checkpoint, str(int(time.time())))
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--run", type=int, default=0, required=False, help="Experiment repetition")
    parser.add_argument("--evaluations", type=int, default=1, required=False, help="Evaluations")
    parser.add_argument("--scenario_id", type=str, default='scenario_5', required=False, help="Scenario ID")
    parser.add_argument('--npc_fixed', type=str_to_bool, default=True, help='npc_fixed')
    parser.add_argument("--npc_json_file", type=str, default=f'npc_initial_location_scenario_5.json', required=False, help="npc_json_file")

    parser.add_argument("--algorithm", type=str, default='random', required=False, help="algorithm action_replay")
    parser.add_argument("--objective", type=str, default='distance+time_to_collision+completion+comfort+speed_diff', required=False, help="objective")
    parser.add_argument("--optuna", type=str_to_bool, default=False, required=False, help="optuna")
    parser.add_argument("--eval", type=str_to_bool, default=False, required=False, help="eval")

    parser.add_argument("--initial_num_episodes", type=int, default=50, required=False, help="initial_num_episodes")
    parser.add_argument("--initial_n_trials", type=int, default=20, required=False, help="initial_n_trials")
    parser.add_argument("--final_num_episodes", type=int, default=200, required=False, help="final_num_episodes")
    parser.add_argument("--final_n_trials", type=int, default=10, required=False, help="final_n_trials")

    parser.add_argument("--lr", type=float, default=0.0001, required=False, help="lr")
    parser.add_argument("--batch_size", type=int, default=16, required=False, help="batch_size")
    parser.add_argument("--learning_starts", type=int, default=512, required=False, help="learning_starts")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, required=False, help="max_grad_norm")
    parser.add_argument("--num_sample_w", type=int, default=16, required=False, help="num_sample_w")
    parser.add_argument("--decay_steps", type=int_none, default=None, required=False, help="decay_steps")
    parser.add_argument("--memory", type=int, default=2000, required=False, help="memory")
    parser.add_argument("--drop_rate", type=float, default=0.0, required=False, help="drop_rate")

    arguments = parser.parse_args()

    os.system('kill $(lsof -t -i:{})'.format(int(arguments.port)))
    os.system('kill $(lsof -t -i:{})'.format(int(arguments.trafficManagerPort)))
    time.sleep(10)
    subprocess.Popen(
        ['cd {}/carla/ && DISPLAY= ./CarlaUE4.sh --world-port={} -opengl'.format(PATH, int(arguments.port))],
        stdout=subprocess.PIPE, universal_newlines=True, shell=True)

    # logs = 'sbatch strategy_ga.slurm --port={} --trafficManagerPort={} --run={} --evaluations={}'.format(
    #     int(arguments.port), int(arguments.trafficManagerPort), int(arguments.run), int(arguments.evaluations))
    logs = f'''sbatch slurm
    --port={arguments.port}
    --trafficManagerPort={arguments.trafficManagerPort}
    --evaluations={arguments.evaluations}
    --scenario_id={arguments.scenario_id}
    --npc_fixed={arguments.npc_fixed}
    --npc_json_file={arguments.npc_json_file}
    --algorithm={arguments.algorithm}
    --objective={arguments.objective}
    --optuna={arguments.optuna}
    --eval={arguments.eval}
    --lr={arguments.lr}
    --batch_size={arguments.batch_size}
    --learning_starts={arguments.learning_starts}
    --max_grad_norm={arguments.max_grad_norm}
    --num_sample_w={arguments.num_sample_w}
    --decay_steps={arguments.decay_steps}
    --memory={arguments.memory}
    --drop_rate={arguments.drop_rate}
    --{arguments.scenarios.split('/')[-1].split('.')[0]}'''
    print('===================================')
    print(logs)
    print('===================================')
    f = open('./logs.md', mode='a', encoding='utf-8')
    f.writelines(logs + '\n')

    time.sleep(20)
    # try:
    main(arguments)
    # except:
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.port)))
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.trafficManagerPort)))
    # finally:
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.port)))
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.trafficManagerPort)))
