#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function

import math
import random
import signal
import sys
import time

import numpy as np
import py_trees
import carla
import torch

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeNoiseParameters
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

from RL.MORL_morl.weights import random_weights


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """


    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.count_tick = None
        self.tick_for_time_step = 40

        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, npc_objects, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.npc_objects = npc_objects
        self.repetition_number = rep_number

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def angle_diff_deg(self, a, b):
        diff = a - b
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff

    def observe(self):
        state = []

        ego_trans = self.ego_vehicles[0].get_transform()
        ego_velocity = self.ego_vehicles[0].get_velocity()
        ego_acc = self.ego_vehicles[0].get_acceleration()
        ego_av = self.ego_vehicles[0].get_angular_velocity()
        ego_state = [
            ego_trans.location.x, ego_trans.location.y,
            ego_trans.rotation.yaw,
            ego_velocity.x, ego_velocity.y,
            ego_acc.x, ego_acc.y,
            ego_av.x, ego_av.y, ego_av.z
        ]
        state += ego_state
        # print(f"ego_state {ego_state}")

        for npc_object in self.npc_objects:
            npc_trans = npc_object.npc_vehicle.get_transform()
            npc_velocity = npc_object.npc_vehicle.get_velocity()
            npc_acc = npc_object.npc_vehicle.get_acceleration()
            npc_av = npc_object.npc_vehicle.get_angular_velocity()
            npc_state = [
                npc_trans.location.x, npc_trans.location.y,
                npc_trans.rotation.yaw,
                npc_velocity.x, npc_velocity.y,
                npc_acc.x, npc_acc.y,
                npc_av.x, npc_av.y, npc_av.z
            ]
            state += npc_state
            # print(f"npc_state {npc_state}")

            dx = npc_trans.location.x - ego_trans.location.x
            dy = npc_trans.location.y - ego_trans.location.y
            relative_distance = math.hypot(dx, dy)

            relative_yaw = self.angle_diff_deg(npc_trans.rotation.yaw, ego_trans.rotation.yaw)

            dvx = npc_velocity.x - ego_velocity.x
            dvy = npc_velocity.y - ego_velocity.y
            relative_velocity = math.hypot(dvx, dvy)

            dax = npc_acc.x - ego_acc.x
            day = npc_acc.y - ego_acc.y
            relative_acceleration = math.hypot(dax, day)

            davx = npc_av.x - ego_av.x
            davy = npc_av.y - ego_av.y
            davz = npc_av.z - ego_av.z
            relative_angular_velocity = math.sqrt(davx ** 2 + davy ** 2 + davz ** 2)

            relative_state = [
                relative_distance,
                relative_yaw,
                relative_velocity,
                relative_acceleration,
                relative_angular_velocity
            ]
            state += relative_state
            # print(f"relative_state {relative_state}")

        return state

    def scenario_tick_json(self, tick):
        tick_data = {f"tick_{tick}": {}}
        ego_trans = self.ego_vehicles[0].get_transform()
        ego_velocity = self.ego_vehicles[0].get_velocity()
        ego_acc = self.ego_vehicles[0].get_acceleration()
        ego_av = self.ego_vehicles[0].get_angular_velocity()
        tick_data[f"tick_{tick}"]["Ego"] = {
            "location": {
                "x": ego_trans.location.x,
                "y": ego_trans.location.y,
                "z": ego_trans.location.z
            },
            "rotation": {
                "pitch": ego_trans.rotation.pitch,
                "yaw": ego_trans.rotation.yaw,
                "roll": ego_trans.rotation.roll
            },
            "velocity": {
                "x": ego_velocity.x,
                "y": ego_velocity.y,
                "z": ego_velocity.z
            },
            "acceleration": {
                "x": ego_acc.x,
                "y": ego_acc.y,
                "z": ego_acc.z
            },
            "angular_velocity": {
                "x": ego_av.x,
                "y": ego_av.y,
                "z": ego_av.z
            }
        }
        npc_num = 0
        for id, actor in CarlaDataProvider.get_actors():
            if actor.id == self.ego_vehicles[0].id:
                continue
            if 'vehicle' in actor.type_id:
                trans = actor.get_transform()
                velocity = actor.get_velocity()
                acc = actor.get_acceleration()
                av = actor.get_angular_velocity()
                tick_data[f"tick_{tick}"][f"NPC_{npc_num}"] = {
                    "id": actor.id,
                    "type": "vehicle",
                    "location": {
                        "x": trans.location.x,
                        "y": trans.location.y,
                        "z": trans.location.z
                    },
                    "rotation": {
                        "pitch": trans.rotation.pitch,
                        "yaw": trans.rotation.yaw,
                        "roll": trans.rotation.roll
                    },
                    "velocity": {
                        "x": velocity.x,
                        "y": velocity.y,
                        "z": velocity.z
                    },
                    "acceleration": {
                        "x": acc.x,
                        "y": acc.y,
                        "z": acc.z
                    },
                    "angular_velocity": {
                        "x": av.x,
                        "y": av.y,
                        "z": av.z
                    }
                }
            elif 'walker' in actor.type_id:
                trans = actor.get_transform()
                velocity = actor.get_velocity()
                tick_data[f"tick_{tick}"][f"NPC_{npc_num}"] = {
                    "id": actor.id,
                    "type": "pedestrian",
                    "location": {
                        "x": trans.location.x,
                        "y": trans.location.y,
                        "z": trans.location.z
                    },
                    "rotation": {
                        "pitch": trans.rotation.pitch,
                        "yaw": trans.rotation.yaw,
                        "roll": trans.rotation.roll
                    },
                    "velocity": {
                        "x": velocity.x,
                        "y": velocity.y,
                        "z": velocity.z
                    }
                }
            npc_num += 1

        return tick_data

    def run_scenario(self):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp)

    def run_initial_scenario(self, scenario_id):

        self._watchdog.start()
        self._running = True
        self.count_tick = 0

        scenario_initial_num = {
            "scenario_1": 40,
            "scenario_2": 40,
            "scenario_3": 40,
            "scenario_4": 40,
            "scenario_5": 40,
            "scenario_6": 40
        }
        for _ in range(scenario_initial_num[scenario_id]):
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                self._tick_scenario(timestamp, scenario_id, -1)

    def run_scenario_envelope(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("envelope")
        self.run_initial_scenario(scenario_id)

        self._running = True
        self.count_tick = 0

        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        pre_percentage_route_completed = 0
        step = 0
        total_reward = 0

        state = self.observe()
        w = random_weights(agent.reward_dim, 1, dist="gaussian", rng=agent.np_random)
        tensor_w = torch.tensor(w).float().to(agent.device)

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                if agent.global_step < agent.learning_starts:
                    action = int(np.random.choice(agent.action_dim, 1)[0])
                else:
                    action = agent.act(torch.as_tensor(state).float().to(agent.device), tensor_w)
                print(f"envelope action {action}")
                for npc_object in self.npc_objects:
                    npc_object.control_flag = False

                min_dis_npc = 1000
                min_ttc_npc = 1000
                max_uncomfort = 0
                acceleration_p = self.calculate_acceleration()
                mean_min_speed = 0
                mean_max_speed = 0
                ego_actor_speed = 0
                speed_points = 0
                for i in range(self.tick_for_time_step):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    if timestamp:
                        self._tick_scenario(timestamp, scenario_id, action)
                        min_dis_npc = min(self.calculate_distance(), min_dis_npc)
                        min_ttc_npc = min(self.calculate_TTC(), min_ttc_npc)
                        frame_mean_speed, max_speed, ego_speed = self.get_speed()
                        mean_min_speed += frame_mean_speed
                        mean_max_speed += max_speed
                        ego_actor_speed += ego_speed
                        speed_points += 1
                        if self.count_tick >= 200:
                            self._running = False
                        if (i + 1) % 5 == 0 or not self._running:
                            acceleration_n = self.calculate_acceleration()
                            count = 5 if (i + 1) % 5 == 0 else (i + 1) % 5
                            max_uncomfort = max(abs(acceleration_n - acceleration_p) / count, max_uncomfort)
                            acceleration_p = acceleration_n
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        pre_percentage_route_completed = criterion.percentage_route_completed

                mean_min_speed /= speed_points
                RATIO = 0.8
                mean_min_speed *= RATIO
                mean_max_speed /= speed_points
                ego_actor_speed /= speed_points
                if ego_actor_speed < mean_min_speed:
                    speed_diff = mean_min_speed - ego_actor_speed
                elif ego_actor_speed > mean_max_speed:
                    speed_diff = ego_actor_speed - mean_max_speed
                else:
                    speed_diff = 0

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                reward_distance = 1 - (np.log(min_dis_npc + 1) / np.log(40)) if not collision else 10
                # reward_distance = 1 - min_dis_npc / (min_dis_npc + 1) if not collision else 1
                reward_ttc = 1 - np.log(np.minimum(min_ttc_npc, 100) + 1) / np.log(100 + 1) if not collision else 10
                reward_completion = 1 - route_completed_percentage / 50 if route_completed_percentage != 0 else 0
                reward_comfort = max_uncomfort / (max_uncomfort + 1)
                reward_speed_diff = speed_diff / np.sqrt(1 + speed_diff ** 2)
                reward_dict = {
                    "distance": reward_distance,
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion,
                    "comfort": reward_comfort,
                    "speed_diff": reward_speed_diff
                }
                reward = [reward_dict[objective] for objective in agent.multi_objective]
                done = collision or not self._running

                total_reward += sum(reward_dict[obj] * (1 / len(agent.multi_objective)) for obj in agent.multi_objective)
                # total_reward += reward_dict[agent.multi_objective[0]] * 0.5 + reward_dict[agent.multi_objective[1]] * 0.5

                next_state = new_state

                agent.global_step += 1
                step += 1

                # Store the transition in memory
                agent.replay_buffer.add(state, action, reward, next_state, done)

                # Perform one step of the optimization (on the policy network)
                if agent.global_step >= agent.learning_starts and agent.replay_buffer.size >= agent.batch_size:
                    critic_losses = agent.update()
                else:
                    critic_losses = [100]

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "distance": min_dis_npc,
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage,
                    "comfort": max_uncomfort,
                    "speed_diff": speed_diff
                }
                agent.write_to_file(
                    [episode, step - 1, action] +
                    [value_dict[obj] for obj in agent.multi_objective] +
                    [collision, collision_type] +
                    [reward_dict[obj] for obj in agent.multi_objective] +
                    [total_reward, critic_losses[0], done, self.count_tick, scenario_duration_system,
                     scenario_duration_game]
                )

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        print('number of ticks: ', self.count_tick)

        return total_reward

    def run_scenario_envelope_eval(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("envelope eval")
        self.run_initial_scenario(scenario_id)

        self._running = True
        self.count_tick = 0

        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        pre_percentage_route_completed = 0
        step = 0
        total_reward = 0

        all_tick_scenario = {f"episode_{episode}": {}}

        state = self.observe()
        w = random_weights(agent.reward_dim, 1, dist="gaussian", rng=agent.np_random)
        tensor_w = torch.tensor(w).float().to(agent.device)

        total_reward_w = np.zeros_like(w)
        gamma = 1.0

        hq = agent.predict(state, tensor_w)
        hq_np = hq.detach().cpu().numpy()[0]
        hq_0 = agent.predict(None, tensor_w)
        hq_np_0 = hq_0.detach().cpu().numpy()[0]

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                action = agent.act(torch.as_tensor(state).float().to(agent.device), tensor_w)
                print(f"envelope eval action {action}")
                for npc_object in self.npc_objects:
                    npc_object.control_flag = False

                all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))

                min_dis_npc = 1000
                min_ttc_npc = 1000
                max_uncomfort = 0
                acceleration_p = self.calculate_acceleration()
                mean_min_speed = 0
                mean_max_speed = 0
                ego_actor_speed = 0
                speed_points = 0
                for i in range(self.tick_for_time_step):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    if timestamp:
                        self._tick_scenario(timestamp, scenario_id, action)
                        min_dis_npc = min(self.calculate_distance(), min_dis_npc)
                        min_ttc_npc = min(self.calculate_TTC(), min_ttc_npc)
                        frame_mean_speed, max_speed, ego_speed = self.get_speed()
                        mean_min_speed += frame_mean_speed
                        mean_max_speed += max_speed
                        ego_actor_speed += ego_speed
                        speed_points += 1
                        if self.count_tick >= 200:
                            self._running = False
                        if (i + 1) % 5 == 0 or not self._running:
                            acceleration_n = self.calculate_acceleration()
                            count = 5 if (i + 1) % 5 == 0 else (i + 1) % 5
                            max_uncomfort = max(abs(acceleration_n - acceleration_p) / count, max_uncomfort)
                            acceleration_p = acceleration_n
                        if self.count_tick % 2 == 0 or not self._running:
                            all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        pre_percentage_route_completed = criterion.percentage_route_completed

                mean_min_speed /= speed_points
                RATIO = 0.8
                mean_min_speed *= RATIO
                mean_max_speed /= speed_points
                ego_actor_speed /= speed_points
                if ego_actor_speed < mean_min_speed:
                    speed_diff = mean_min_speed - ego_actor_speed
                elif ego_actor_speed > mean_max_speed:
                    speed_diff = ego_actor_speed - mean_max_speed
                else:
                    speed_diff = 0

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                reward_distance = 1 - (np.log(min_dis_npc + 1) / np.log(40)) if not collision else 10
                # reward_distance = 1 - min_dis_npc / (min_dis_npc + 1) if not collision else 1
                reward_ttc = 1 - np.log(np.minimum(min_ttc_npc, 100) + 1) / np.log(100 + 1) if not collision else 10
                reward_completion = 1 - route_completed_percentage / 50 if route_completed_percentage != 0 else 0
                reward_comfort = max_uncomfort / (max_uncomfort + 1)
                reward_speed_diff = speed_diff / np.sqrt(1 + speed_diff ** 2)
                reward_dict = {
                    "distance": reward_distance,
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion,
                    "comfort": reward_comfort,
                    "speed_diff": reward_speed_diff
                }
                reward = [reward_dict[objective] for objective in agent.multi_objective]
                done = collision or not self._running

                total_reward += sum(
                    reward_dict[obj] * (1 / len(agent.multi_objective)) for obj in agent.multi_objective)
                # total_reward += reward_dict[agent.multi_objective[0]] * 0.5 + reward_dict[agent.multi_objective[1]] * 0.5

                total_reward_w += gamma * np.array(reward)
                gamma *= agent.gamma

                next_state = new_state

                step += 1

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "distance": min_dis_npc,
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage,
                    "comfort": max_uncomfort,
                    "speed_diff": speed_diff
                }
                agent.write_to_file(
                    [episode, step - 1, action] +
                    [value_dict[obj] for obj in agent.multi_objective] +
                    [collision, collision_type] +
                    [reward_dict[obj] for obj in agent.multi_objective] +
                    [total_reward, None, done, self.count_tick, scenario_duration_system,
                     scenario_duration_game]
                )

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        agent.update_all_episode_scenario(all_tick_scenario)
        print('number of ticks: ', self.count_tick)

        agent.write_pareto_to_file(
            [episode] +
            [total_reward_w[i] for i in range(len(agent.multi_objective))] +
            [hq_np[i] for i in range(len(agent.multi_objective))] +
            [hq_np_0[i] for i in range(len(agent.multi_objective))]
        )

        return total_reward

    def run_scenario_deepcollision(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("deepcollision")
        self.run_initial_scenario(scenario_id)

        self._running = True
        self.count_tick = 0

        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        pre_percentage_route_completed = 0
        step = 0
        total_reward = 0

        state = self.observe()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                if agent.global_step < agent.learning_starts:
                    action = torch.tensor([[random.randrange(agent.action_dim)]], device=agent.device, dtype=torch.long)
                else:
                    action = agent.act(state)
                print(f"deepcollision {action}")
                for npc_object in self.npc_objects:
                    npc_object.control_flag = False

                min_dis_npc = 1000
                min_ttc_npc = 1000
                max_uncomfort = 0
                acceleration_p = self.calculate_acceleration()
                mean_min_speed = 0
                mean_max_speed = 0
                ego_actor_speed = 0
                speed_points = 0
                for i in range(self.tick_for_time_step):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    if timestamp:
                        self._tick_scenario(timestamp, scenario_id, action)
                        min_dis_npc = min(self.calculate_distance(), min_dis_npc)
                        min_ttc_npc = min(self.calculate_TTC(), min_ttc_npc)
                        frame_mean_speed, max_speed, ego_speed = self.get_speed()
                        mean_min_speed += frame_mean_speed
                        mean_max_speed += max_speed
                        ego_actor_speed += ego_speed
                        speed_points += 1
                        if self.count_tick >= 200:
                            self._running = False
                        if (i + 1) % 5 == 0 or not self._running:
                            acceleration_n = self.calculate_acceleration()
                            count = 5 if (i + 1) % 5 == 0 else (i + 1) % 5
                            max_uncomfort = max(abs(acceleration_n - acceleration_p) / count, max_uncomfort)
                            acceleration_p = acceleration_n
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        pre_percentage_route_completed = criterion.percentage_route_completed

                mean_min_speed /= speed_points
                RATIO = 0.8
                mean_min_speed *= RATIO
                mean_max_speed /= speed_points
                ego_actor_speed /= speed_points
                if ego_actor_speed < mean_min_speed:
                    speed_diff = mean_min_speed - ego_actor_speed
                elif ego_actor_speed > mean_max_speed:
                    speed_diff = ego_actor_speed - mean_max_speed
                else:
                    speed_diff = 0

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                reward_distance = 1 - (np.log(min_dis_npc + 1) / np.log(40)) if not collision else 10
                reward_ttc = 1 - np.log(np.minimum(min_ttc_npc, 100) + 1) / np.log(100 + 1) if not collision else 10
                reward_completion = 1 - route_completed_percentage / 50 if route_completed_percentage != 0 else 0
                reward_comfort = max_uncomfort / (max_uncomfort + 1)
                reward_speed_diff = speed_diff / np.sqrt(1 + speed_diff ** 2)
                reward_dict = {
                    "distance": reward_distance,
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion,
                    "comfort": reward_comfort,
                    "speed_diff": reward_speed_diff
                }
                reward = sum(reward_dict[obj] * (1 / len(agent.single_objective)) for obj in agent.single_objective)
                # if agent.single_objective[0] == agent.single_objective[1]:
                #     reward = reward_dict[agent.single_objective[0]]
                # else:
                #     reward = reward_dict[agent.single_objective[0]] * 0.5 + reward_dict[agent.single_objective[1]] * 0.5
                done = collision or not self._running

                total_reward += reward

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(new_state, dtype=torch.float32, device=agent.device).unsqueeze(0)

                agent.global_step += 1
                step += 1

                # Store the transition in memory
                agent.replay_buffer.push(state, action, next_state, torch.tensor([reward], device=agent.device))

                # Perform one step of the optimization (on the policy network)
                if agent.global_step >= agent.learning_starts:
                    critic_losses = agent.update()
                else:
                    critic_losses = [100]

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "distance": min_dis_npc,
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage,
                    "comfort": max_uncomfort,
                    "speed_diff": speed_diff
                }
                agent.write_to_file(
                    [episode, step - 1, action] +
                    [value_dict[obj] for obj in agent.single_objective] +
                    [collision, collision_type] +
                    [reward_dict[obj] for obj in agent.single_objective] +
                    [total_reward, critic_losses[0], done, self.count_tick, scenario_duration_system,
                     scenario_duration_game]
                )

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        print('number of ticks: ', self.count_tick)

        return total_reward

    def run_scenario_deepcollision_eval(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("deepcollision eval")
        self.run_initial_scenario(scenario_id)

        self._running = True
        self.count_tick = 0

        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        pre_percentage_route_completed = 0
        step = 0
        total_reward = 0

        all_tick_scenario = {f"episode_{episode}": {}}

        state = self.observe()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        total_reward_w = np.zeros(len(agent.single_objective))
        gamma = 1.0

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                action = agent.act(state)
                print(f"deepcollision eval {action}")
                for npc_object in self.npc_objects:
                    npc_object.control_flag = False

                all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))

                min_dis_npc = 1000
                min_ttc_npc = 1000
                max_uncomfort = 0
                acceleration_p = self.calculate_acceleration()
                mean_min_speed = 0
                mean_max_speed = 0
                ego_actor_speed = 0
                speed_points = 0
                for i in range(self.tick_for_time_step):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    if timestamp:
                        self._tick_scenario(timestamp, scenario_id, action)
                        min_dis_npc = min(self.calculate_distance(), min_dis_npc)
                        min_ttc_npc = min(self.calculate_TTC(), min_ttc_npc)
                        frame_mean_speed, max_speed, ego_speed = self.get_speed()
                        mean_min_speed += frame_mean_speed
                        mean_max_speed += max_speed
                        ego_actor_speed += ego_speed
                        speed_points += 1
                        if self.count_tick >= 200:
                            self._running = False
                        if (i + 1) % 5 == 0 or not self._running:
                            acceleration_n = self.calculate_acceleration()
                            count = 5 if (i + 1) % 5 == 0 else (i + 1) % 5
                            max_uncomfort = max(abs(acceleration_n - acceleration_p) / count, max_uncomfort)
                            acceleration_p = acceleration_n
                        if self.count_tick % 2 == 0 or not self._running:
                            all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        pre_percentage_route_completed = criterion.percentage_route_completed

                mean_min_speed /= speed_points
                RATIO = 0.8
                mean_min_speed *= RATIO
                mean_max_speed /= speed_points
                ego_actor_speed /= speed_points
                if ego_actor_speed < mean_min_speed:
                    speed_diff = mean_min_speed - ego_actor_speed
                elif ego_actor_speed > mean_max_speed:
                    speed_diff = ego_actor_speed - mean_max_speed
                else:
                    speed_diff = 0

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                reward_distance = 1 - (np.log(min_dis_npc + 1) / np.log(40)) if not collision else 10
                reward_ttc = 1 - np.log(np.minimum(min_ttc_npc, 100) + 1) / np.log(100 + 1) if not collision else 10
                reward_completion = 1 - route_completed_percentage / 50 if route_completed_percentage != 0 else 0
                reward_comfort = max_uncomfort / (max_uncomfort + 1)
                reward_speed_diff = speed_diff / np.sqrt(1 + speed_diff ** 2)
                reward_dict = {
                    "distance": reward_distance,
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion,
                    "comfort": reward_comfort,
                    "speed_diff": reward_speed_diff
                }
                reward = sum(reward_dict[obj] * (1 / len(agent.single_objective)) for obj in agent.single_objective)
                # if agent.single_objective[0] == agent.single_objective[1]:
                #     reward = reward_dict[agent.single_objective[0]]
                # else:
                #     reward = reward_dict[agent.single_objective[0]] * 0.5 + reward_dict[agent.single_objective[1]] * 0.5
                done = collision or not self._running

                total_reward += reward

                total_reward_w += gamma * np.array([reward_dict[objective] for objective in agent.single_objective])
                gamma *= agent.gamma

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(new_state, dtype=torch.float32, device=agent.device).unsqueeze(0)

                step += 1

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "distance": min_dis_npc,
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage,
                    "comfort": max_uncomfort,
                    "speed_diff": speed_diff
                }
                agent.write_to_file(
                    [episode, step - 1, action] +
                    [value_dict[obj] for obj in agent.single_objective] +
                    [collision, collision_type] +
                    [reward_dict[obj] for obj in agent.single_objective] +
                    [total_reward, None, done, self.count_tick, scenario_duration_system,
                     scenario_duration_game]
                )

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        agent.update_all_episode_scenario(all_tick_scenario)
        print('number of ticks: ', self.count_tick)

        agent.write_pareto_to_file(
            [episode] +
            [total_reward_w[i] for i in range(len(agent.single_objective))]
        )

        return total_reward

    def run_scenario_random(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("random")
        self.run_initial_scenario(scenario_id)

        self._running = True
        self.count_tick = 0

        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        pre_percentage_route_completed = 0
        step = 0
        total_reward = 0

        all_tick_scenario = {f"episode_{episode}": {}}

        total_reward_w = np.zeros(len(agent.objective))
        gamma = 1.0

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                action = int(np.random.choice(agent.action_dim, 1)[0])
                print(f"random action {action}")
                for npc_object in self.npc_objects:
                    npc_object.control_flag = False

                all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))

                min_dis_npc = 1000
                min_ttc_npc = 1000
                max_uncomfort = 0
                acceleration_p = self.calculate_acceleration()
                mean_min_speed = 0
                mean_max_speed = 0
                ego_actor_speed = 0
                speed_points = 0
                for i in range(self.tick_for_time_step):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    if timestamp:
                        self._tick_scenario(timestamp, scenario_id, action)
                        min_dis_npc = min(self.calculate_distance(), min_dis_npc)
                        min_ttc_npc = min(self.calculate_TTC(), min_ttc_npc)
                        frame_mean_speed, max_speed, ego_speed = self.get_speed()
                        mean_min_speed += frame_mean_speed
                        mean_max_speed += max_speed
                        ego_actor_speed += ego_speed
                        speed_points += 1
                        if self.count_tick >= 200:
                            self._running = False
                        if (i + 1) % 5 == 0 or not self._running:
                            acceleration_n = self.calculate_acceleration()
                            count = 5 if (i + 1) % 5 == 0 else (i + 1) % 5
                            max_uncomfort = max(abs(acceleration_n - acceleration_p) / count, max_uncomfort)
                            acceleration_p = acceleration_n
                        if self.count_tick % 2 == 0 or not self._running:
                            all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        pre_percentage_route_completed = criterion.percentage_route_completed

                mean_min_speed /= speed_points
                RATIO = 0.8
                mean_min_speed *= RATIO
                mean_max_speed /= speed_points
                ego_actor_speed /= speed_points
                if ego_actor_speed < mean_min_speed:
                    speed_diff = mean_min_speed - ego_actor_speed
                elif ego_actor_speed > mean_max_speed:
                    speed_diff = ego_actor_speed - mean_max_speed
                else:
                    speed_diff = 0

                collision, collision_type = self.analyze_collision_step()
                reward_distance = 1 - (np.log(min_dis_npc + 1) / np.log(40)) if not collision else 10
                reward_ttc = 1 - np.log(np.minimum(min_ttc_npc, 100) + 1) / np.log(100 + 1) if not collision else 10
                reward_completion = 1 - route_completed_percentage / 50 if route_completed_percentage != 0 else 0
                reward_comfort = max_uncomfort / (max_uncomfort + 1)
                reward_speed_diff = speed_diff / np.sqrt(1 + speed_diff ** 2)
                done = collision or not self._running

                reward_dict = {
                    "distance": reward_distance,
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion,
                    "comfort": reward_comfort,
                    "speed_diff": reward_speed_diff
                }
                total_reward += sum(reward_dict[obj] * (1 / len(agent.objective)) for obj in agent.objective)

                total_reward_w += gamma * np.array([reward_dict[objective] for objective in agent.objective])
                gamma *= agent.gamma

                step += 1

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "distance": min_dis_npc,
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage,
                    "comfort": max_uncomfort,
                    "speed_diff": speed_diff
                }
                agent.write_to_file(
                    [episode, step - 1, action] +
                    [value_dict[obj] for obj in agent.objective] +
                    [collision, collision_type] +
                    [reward_dict[obj] for obj in agent.objective] +
                    [total_reward, None, done, self.count_tick, scenario_duration_system,
                     scenario_duration_game]
                )

                if done:
                    self._running = False

        agent.update_all_episode_scenario(all_tick_scenario)
        print('number of ticks: ', self.count_tick)

        agent.write_pareto_to_file(
            [episode] +
            [total_reward_w[i] for i in range(len(agent.objective))]
        )

        return None

    def run_scenario_action_replay(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("action_replay")
        self.run_initial_scenario(scenario_id)

        self._running = True
        self.count_tick = 0

        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        pre_percentage_route_completed = 0
        step = 0
        total_reward = 0

        # all_tick_scenario = {f"episode_{episode}": {}}

        total_reward_w = np.zeros(len(agent.objective))
        gamma = 1.0

        action_list = [5, 5, 0, 4, 11]

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                action = action_list[step]
                print(f"action_replay action {action}")
                for npc_object in self.npc_objects:
                    npc_object.control_flag = False

                # all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))

                min_dis_npc = 1000
                min_ttc_npc = 1000
                max_uncomfort = 0
                acceleration_p = self.calculate_acceleration()
                mean_min_speed = 0
                mean_max_speed = 0
                ego_actor_speed = 0
                speed_points = 0
                for i in range(self.tick_for_time_step):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    if timestamp:
                        self._tick_scenario(timestamp, scenario_id, action)
                        min_dis_npc = min(self.calculate_distance(), min_dis_npc)
                        min_ttc_npc = min(self.calculate_TTC(), min_ttc_npc)
                        frame_mean_speed, max_speed, ego_speed = self.get_speed()
                        mean_min_speed += frame_mean_speed
                        mean_max_speed += max_speed
                        ego_actor_speed += ego_speed
                        speed_points += 1
                        if self.count_tick >= 200:
                            self._running = False
                        if (i + 1) % 5 == 0 or not self._running:
                            acceleration_n = self.calculate_acceleration()
                            count = 5 if (i + 1) % 5 == 0 else (i + 1) % 5
                            max_uncomfort = max(abs(acceleration_n - acceleration_p) / count, max_uncomfort)
                            acceleration_p = acceleration_n
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        pre_percentage_route_completed = criterion.percentage_route_completed

                mean_min_speed /= speed_points
                RATIO = 0.8
                mean_min_speed *= RATIO
                mean_max_speed /= speed_points
                ego_actor_speed /= speed_points
                if ego_actor_speed < mean_min_speed:
                    speed_diff = mean_min_speed - ego_actor_speed
                elif ego_actor_speed > mean_max_speed:
                    speed_diff = ego_actor_speed - mean_max_speed
                else:
                    speed_diff = 0

                collision, collision_type = self.analyze_collision_step()
                reward_distance = 1 - (np.log(min_dis_npc + 1) / np.log(40)) if not collision else 10
                reward_ttc = 1 - np.log(np.minimum(min_ttc_npc, 100) + 1) / np.log(100 + 1) if not collision else 10
                reward_completion = 1 - route_completed_percentage / 50 if route_completed_percentage != 0 else 0
                reward_comfort = max_uncomfort / (max_uncomfort + 1)
                reward_speed_diff = speed_diff / np.sqrt(1 + speed_diff ** 2)
                done = collision or not self._running

                reward_dict = {
                    "distance": reward_distance,
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion,
                    "comfort": reward_comfort,
                    "speed_diff": reward_speed_diff
                }
                total_reward += sum(reward_dict[obj] * (1 / len(agent.objective)) for obj in agent.objective)
                # if agent.objective[0] == agent.objective[1]:
                #     total_reward += reward_dict[agent.objective[0]]
                # else:
                #     total_reward += reward_dict[agent.objective[0]] * 0.5 + reward_dict[agent.objective[1]] * 0.5

                total_reward_w += gamma * np.array([reward_dict[objective] for objective in agent.objective])
                gamma *= agent.gamma

                step += 1

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "distance": min_dis_npc,
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage,
                    "comfort": max_uncomfort,
                    "speed_diff": speed_diff
                }
                agent.write_to_file(
                    [episode, step - 1, action] +
                    [value_dict[obj] for obj in agent.objective] +
                    [collision, collision_type] +
                    [reward_dict[obj] for obj in agent.objective] +
                    [total_reward, None, done, self.count_tick, scenario_duration_system,
                     scenario_duration_game]
                )

                if done:
                    self._running = False

        # agent.update_all_episode_scenario(all_tick_scenario)
        print('number of ticks: ', self.count_tick)

        agent.write_pareto_to_file(
            [episode] +
            [total_reward_w[i] for i in range(len(agent.objective))]
        )

        return None

    def _tick_scenario(self, timestamp, scenario_id, action):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            self.count_tick += 1

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                agent_out = self._agent()
                ego_action = agent_out[0]

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            if 50 < self.count_tick < 100 and scenario_id in ['scenario_6']:
                ego_action = self.change_control(ego_action)

            self.ego_vehicles[0].apply_control(ego_action)

            # NPC Vehicle
            if action != -1:
                if action < 6:
                    self.npc_objects[0].exe_action(action)
                else:
                    self.npc_objects[1].exe_action(action - 6)

            # judge_stop_walker
            actors = CarlaDataProvider.get_actors()
            for actor_id, actor in actors:
                if 'walker' in actor.type_id:
                    self.judge_stop_walker(actor)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=40),
                                                    carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        return

    def get_speed(self, max_speed=5):
        all_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle*')
        frame_mean_speed = 0
        for vehicle in all_vehicles:
            if vehicle.id == self.ego_vehicles[0].id:
                continue
            npc_speed = CarlaDataProvider.get_velocity(vehicle)
            frame_mean_speed += npc_speed if npc_speed <= max_speed else max_speed
        frame_mean_speed /= len(all_vehicles) - 1
        ego_speed = CarlaDataProvider.get_velocity(self.ego_vehicles[0])

        return frame_mean_speed, max_speed, ego_speed

    def calculate_acceleration(self):
        acceleration_squared = self.ego_vehicles[0].get_acceleration().x ** 2
        acceleration_squared += self.ego_vehicles[0].get_acceleration().y ** 2
        return math.sqrt(acceleration_squared)

    def calculate_speed(self, velocity):
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2)

    def get_distance(self, actor, x, y):
        return math.sqrt((actor.get_location().x - x) ** 2 + (actor.get_location().y - y) ** 2)

    def get_line_y_x(self, actor):
        actor_location_x = actor.get_location().x
        actor_location_y = actor.get_location().y

        actor_velocity_x = actor.get_velocity().x if actor.get_velocity().x > 0.01 else 0.0
        actor_velocity_y = 0.0001 if actor.get_velocity().y == 0.0 else actor.get_velocity().y if actor.get_velocity().y > 0.01 else 0.0001

        return actor_velocity_x / actor_velocity_y, actor_location_x - (actor_velocity_x / actor_velocity_y) * actor_location_y

    def judge_same_line_y_x(self, actor1, actor2, k1, k2):
        judge = False
        direction_vector = (actor1.get_location().y - actor2.get_location().y,
                            actor1.get_location().x - actor2.get_location().x)
        distance = self.get_distance(actor1, actor2.get_location().x, actor2.get_location().y)

        if abs(k1 - k2) < 0.2:
            if abs((actor1.get_location().x - actor2.get_location().x) /
                   ((actor1.get_location().y - actor2.get_location().y) if (actor1.get_location().y - actor2.get_location().y) != 0 else 0.0001)
                   - (k1 + k2) / 2) < 0.05:
                judge = True

        if not judge:
            return judge, 100000

        actor1_velocity = actor1.get_velocity()
        actor2_velocity = actor2.get_velocity()
        actor1_speed = self.calculate_speed(actor1_velocity)
        actor2_speed = self.calculate_speed(actor2_velocity)
        if direction_vector[0] * actor1_velocity.y >= 0 and direction_vector[1] * actor1_velocity.x >= 0:
            TTC = distance / ((actor2_speed - actor1_speed) if (actor2_speed - actor1_speed) != 0 else 0.0001)
        else:
            TTC = distance / ((actor1_speed - actor2_speed) if (actor1_speed - actor2_speed) != 0 else 0.0001)
        if TTC < 0:
            TTC = 100000

        return judge, TTC

    def calculate_TTC(self):
        trajectory_ego_k, trajectory_ego_b = self.get_line_y_x(self.ego_vehicles[0])
        ego_speed = self.calculate_speed(self.ego_vehicles[0].get_velocity())
        ego_speed = ego_speed if ego_speed > 0.01 else 0.01

        actors = CarlaDataProvider.get_actors()
        TTC = 100000

        for actor_id, actor in actors:
            if actor.id == self.ego_vehicles[0].id:
                continue
            trajectory_actor_k, trajectory_actor_b = self.get_line_y_x(actor)
            actor_speed = self.calculate_speed(actor.get_velocity())
            actor_speed = actor_speed if actor_speed > 0.01 else 0.01

            same_lane, ttc = self.judge_same_line_y_x(self.ego_vehicles[0], actor, trajectory_ego_k, trajectory_actor_k)
            if same_lane:
                TTC = min(TTC, ttc)
            else:
                trajectory_ego_k = trajectory_ego_k if trajectory_ego_k != 0 else trajectory_ego_k + 0.0001
                trajectory_actor_k = trajectory_actor_k if trajectory_actor_k != 0 else trajectory_actor_k + 0.0001
                trajectory_actor_k = trajectory_actor_k if (trajectory_ego_k - trajectory_actor_k) != 0 else trajectory_actor_k + 0.0001

                collision_point_y = (trajectory_actor_b - trajectory_ego_b) / (trajectory_ego_k - trajectory_actor_k)
                collision_point_x = ((trajectory_ego_k * trajectory_actor_b - trajectory_actor_k * trajectory_ego_b) /
                                     (trajectory_ego_k - trajectory_actor_k))

                ego_distance = self.get_distance(self.ego_vehicles[0], collision_point_x, collision_point_y)
                actor_distance = self.get_distance(actor, collision_point_x, collision_point_y)
                time_ego = ego_distance / ego_speed
                time_actor = actor_distance / actor_speed
                if ego_speed == 0.01 and actor_speed == 0.01:
                    TTC = min(TTC, 100000)
                else:
                    if abs(time_ego - time_actor) < 1:
                        TTC = min(TTC, (time_ego + time_actor) / 2)
                        # print("collision_point_x", collision_point_x, "collision_point_y", collision_point_y)
                        # print("ego_point", self.ego_vehicles[0].get_location().x, self.ego_vehicles[0].get_location().y)
                        # print("ego_velocity", self.ego_vehicles[0].get_velocity().x, self.ego_vehicles[0].get_velocity().y)
                        # print("actor_point", actor.get_location().x, actor.get_location().y)
                        # print("actor_velocity", actor.get_velocity().x, actor.get_velocity().y)

        return TTC

    def calculate_distance(self):
        actors = CarlaDataProvider.get_actors()
        ego_location = self.ego_vehicles[0].get_location()
        min_dis_npc = 1000

        for actor_id, actor in actors:
            if actor.id == self.ego_vehicles[0].id:
                continue
            actor_location = actor.get_location()
            dis = math.sqrt(
                (ego_location.x - actor_location.x) ** 2 + (ego_location.y - actor_location.y) ** 2)
            min_dis_npc = min(dis, min_dis_npc)

        return min_dis_npc

    @staticmethod
    def judge_stop_walker(walker):
        walker_control = walker.get_control()
        if walker_control.speed != 0:
            actors = CarlaDataProvider.get_actors()
            for actor_id, actor in actors:
                if 'vehicle' in actor.type_id:
                    dis = math.sqrt(
                        (actor.get_location().x - walker.get_location().x) ** 2 + (
                                actor.get_location().y - walker.get_location().y) ** 2)
                    if dis < 5.0:
                        walker_control.speed = 0  # {0.94,1.43} https://www.fhwa.dot.gov/publications/research/safety/pedbike/05085/chapt8.cfm
                        walker_control.jump = False
                        walker.apply_control(walker_control)
                        break

    @staticmethod
    def generate_noise():
        _noise_mean = 0  # Mean value of steering noise
        _noise_std = 0.01  # Std. deviation of steering noise
        _dynamic_mean_for_steer = 0.001
        _dynamic_mean_for_throttle = 0.045
        _abort_distance_to_intersection = 10
        _current_steer_noise = [0]  # This is a list, since lists are mutable
        _current_throttle_noise = [0]
        turn = ChangeNoiseParameters(_current_steer_noise, _current_throttle_noise,
                                     _noise_mean, _noise_std, _dynamic_mean_for_steer,
                                     _dynamic_mean_for_throttle)  # Mean value of steering noise

        turn.update()

        # print(turn._new_steer_noise[0])
        # print(turn._new_throttle_noise[0])
        return turn

    def change_control(self, control):
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.
        """
        turn = self.generate_noise()
        control.steer += turn._new_steer_noise[0]
        control.throttle += turn._new_throttle_noise[0]

        return control

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m' + 'SUCCESS' + '\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        ResultOutputProvider(self, global_result)

    def analyze_collision_step(self):
        collision = False
        collision_type = None
        for node in self.scenario.get_criteria():
            if node.list_traffic_events:
                for event in node.list_traffic_events:
                    if event.get_type() == TrafficEventType.COLLISION_STATIC:
                        collision = True
                        collision_type = 'static'
                    elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                        collision = True
                        collision_type = 'pedestrian'
                    elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                        collision = True
                        collision_type = 'vehicle'
        return collision, collision_type
