import csv
import itertools
import json
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from dtaidistance import dtw_ndim
from matplotlib import pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.stats import mannwhitneyu, spearmanr, fisher_exact
from statsmodels.stats.multitest import multipletests


def create_floder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def tensorboard_smoothing(x, smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x


def hamming_distance(a, b, max_length=5):
    if len(a) != len(b):
        return max_length
    return sum(x != y for x, y in zip(a, b))


def compute_action_difference(action_list):
    diff_sum = 0
    total_pairs = 0
    for i in range(0, len(action_list) - 1):
        for j in range(i + 1, len(action_list)):
            diff = hamming_distance(action_list[i], action_list[j])
            diff_sum += diff
            total_pairs += 1
    action_diff = diff_sum / total_pairs if total_pairs != 0 else 0

    counter = Counter(action_list)
    keys = list(counter.keys())
    diff_sum = 0
    total_pairs = 0
    for i in range(0, len(keys) - 1):
        for j in range(i + 1, len(keys)):
            diff = hamming_distance(keys[i], keys[j])
            diff_sum += diff
            total_pairs += 1
    action_diff_no_num = diff_sum / total_pairs if total_pairs != 0 else 0

    return action_diff, action_diff_no_num


def euclidean_distances(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def angle_diff_deg(a, b):
    return (b - a + 180) % 360 - 180


def normalize_dist(x):
    return x / (x + 1)


def episode_to_pairwise_sequence(episode_data):
    ticks_sorted = sorted(episode_data.keys(), key=lambda x: int(x.split('_')[1]))
    seq = []

    for tick in ticks_sorted:
        tick_data = episode_data[tick]

        def extract(entity):
            return {
                'location': (entity['location']['x'], entity['location']['y']),
                'rotation': (entity['rotation']['yaw'],),
                'velocity': (entity['velocity']['x'], entity['velocity']['y']),
                'acceleration': (entity['acceleration']['x'], entity['acceleration']['y']),
                'angular_velocity': (
                entity['angular_velocity']['x'], entity['angular_velocity']['y'], entity['angular_velocity']['z'])
            }

        ego = extract(tick_data['Ego'])
        npc0 = extract(tick_data['NPC_0'])
        npc1 = extract(tick_data['NPC_1'])

        def compute_all(entity_a, entity_b):
            return [
                normalize_dist(euclidean_distances(entity_a['location'], entity_b['location'])),
                normalize_dist(abs(angle_diff_deg(entity_a['rotation'][0], entity_b['rotation'][0]))),
                normalize_dist(euclidean_distances(entity_a['velocity'], entity_b['velocity'])),
                normalize_dist(euclidean_distances(entity_a['acceleration'], entity_b['acceleration'])),
                normalize_dist(euclidean_distances(entity_a['angular_velocity'], entity_b['angular_velocity'])),
            ]

        tick_metrics = compute_all(ego, npc0) + compute_all(ego, npc1) + compute_all(npc0, npc1)
        seq.append(tick_metrics)

    return seq


def compute_dtw_for_episodes_pairwise(data, episodes):
    total_dist = 0
    count = 0
    for i in range(len(episodes)-1):
        seq_i = np.array(episode_to_pairwise_sequence(data[episodes[i]]))
        for j in range(i+1, len(episodes)):
            seq_j = np.array(episode_to_pairwise_sequence(data[episodes[j]]))
            dist = dtw_ndim.distance(seq_i, seq_j, use_c=True)
            total_dist += dist
            count += 1
    return total_dist / count if count > 0 else 0


def json_compute_dtw_pairwise(folder_path, file_name):
    json_path = f'{folder_path}/{file_name}_all_episode_scenario.json'
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_episodes = list(data.keys())

    all_dtw = compute_dtw_for_episodes_pairwise(data, all_episodes) if len(all_episodes) >= 2 else 0

    # collision episodes
    df = pd.read_csv(f'{folder_path}/{file_name}.csv')
    collision_episodes = []
    for episode in all_episodes:
        episode_rows = df[df['episode'] == int(episode.split('_')[1])]
        if str(episode_rows.iloc[-1]['collision']).lower() == 'true':
            collision_episodes.append(episode)
    collision_dtw = compute_dtw_for_episodes_pairwise(data, collision_episodes) if len(collision_episodes) >= 2 else 0

    return all_dtw, collision_dtw


def normalize_list(data, obj):
    data = np.asarray(data, dtype=float)
    if obj == 'distance':
        normalized = 1 - np.log(data + 1) / np.log(40)
    elif obj == 'time_to_collision':
        normalized = 1 - np.log(np.minimum(data, 100) + 1) / np.log(100 + 1)
    elif obj == 'completion':
        normalized = np.where(data != 0, 1 - data / 100, 0)
    elif obj == 'comfort':
        normalized = data / (data + 1)
    elif obj == 'speed_diff':
        normalized = data / np.sqrt(1 + data ** 2)
    return normalized.tolist()


def converge(data, file_name, folder_path):
    data['episode'] = data['episode'].astype(int)
    episode_start = 0
    episode_end = data['episode'].iloc[-1]
    episode = episode_start

    loss = []
    dis, ttc, comp, comf, spdif = [], [], [], [], []
    rew_dis, rew_ttc, rew_comp, rew_comf, rew_spdif, rew_total = [], [], [], [], [], []
    collision = []
    loss_episode = []
    dis_episode, ttc_episode, comp_episode, comf_episode, spdif_episode = [], [], [], [], []
    reward_dis, reward_ttc, reward_comp, reward_comf, reward_spdif = [], [], [], [], []
    coll = []

    tol_dis, tol_ttc, tol_comp, tol_comf, tol_spdif = [], [], [], [], []
    col_num, dis_num, ttc_num, comp_num, comf_num, spdif_num = 0, 0, 0, 0, 0, 0
    dis_num_dis, ttc_num_ttc, comp_num_comp, comf_num_comf, spdif_num_spdif = [], [], [], [], []
    col_dis, col_ttc, col_comp, col_comf, col_spdif = [], [], [], [], []
    col_comp_num, col_comp_dis, col_comp_ttc, col_comp_comp = 0, [], [], []
    col_comf_num, col_comf_dis, col_comf_ttc, col_comf_comf = 0, [], [], []
    col_spdif_num, col_spdif_dis, col_spdif_ttc, col_spdif_spdif = 0, [], [], []
    dis_ttc_num, dis_ttc_dis, dis_ttc_ttc = 0, [], []
    dis_comp_num, dis_comp_dis, dis_comp_comp = 0, [], []
    dis_comf_num, dis_comf_dis, dis_comf_comf = 0, [], []
    dis_spdif_num, dis_spdif_dis, dis_spdif_spdif = 0, [], []
    ttc_comp_num, ttc_comp_ttc, ttc_comp_comp = 0, [], []
    ttc_comf_num, ttc_comf_ttc, ttc_comf_comf = 0, [], []
    ttc_spdif_num, ttc_spdif_ttc, ttc_spdif_spdif = 0, [], []
    comp_comf_num, comp_comf_comp, comp_comf_comf = 0, [], []
    comp_spdif_num, comp_spdif_comp, comp_spdif_spdif = 0, [], []
    comf_spdif_num, comf_spdif_comf, comf_spdif_spdif = 0, [], []

    action, action_list, col_action_list = [], [], []

    has_loss = 'loss' in data.columns and data['loss'].notnull().any()
    has_dis = 'distance' in data.columns or 'distance' in file_name
    has_ttc = 'time_to_collision' in data.columns or 'time_to_collision' in file_name
    has_comp = 'completion' in data.columns or 'completion' in file_name
    has_comf = 'comfort' in data.columns or 'comfort' in file_name
    has_spdif = 'speed_diff' in data.columns or 'speed_diff' in file_name

    data.index = data.index.astype(int)
    for i in range(data[data['episode'] == episode_start].index.min(),
                   data[data['episode'] == episode_end].index.max() + 2):
        if i == len(data) or data['episode'].iloc[i] != episode:
            if has_loss: loss.append(np.mean(loss_episode))
            if has_dis: dis.append(np.mean(dis_episode))
            if has_ttc: ttc.append(np.mean(ttc_episode))
            if has_comp: comp.append(np.mean(comp_episode))
            if has_comf: comf.append(np.mean(comf_episode))
            if has_spdif: spdif.append(np.mean(spdif_episode))

            if data['episode'].iloc[i - 1] >= episode_end - 99:
                if has_dis:
                    tol_dis.append(np.mean(dis_episode))
                    if np.mean(dis_episode) < 5.0:
                        dis_num += 1
                        dis_num_dis.append(np.mean(dis_episode))
                if has_ttc:
                    tol_ttc.append(np.mean(ttc_episode))
                    if np.mean(ttc_episode) < 1.0:
                        ttc_num += 1
                        ttc_num_ttc.append(np.mean(ttc_episode))
                if has_comp:
                    tol_comp.append(sum(comp_episode))
                    if 100 - sum(comp_episode) > 0.1:
                        comp_num += 1
                        comp_num_comp.append(sum(comp_episode))
                if has_comf:
                    tol_comf.append(np.mean(comf_episode))
                    if np.mean(comf_episode) > 0.9:
                        comf_num += 1
                        comf_num_comf.append(np.mean(comf_episode))
                if has_spdif:
                    tol_spdif.append(np.mean(spdif_episode))
                    if np.mean(spdif_episode) != 0.0:
                        spdif_num += 1
                        spdif_num_spdif.append(np.mean(spdif_episode))
                action_list.append(tuple(action))

                if data['collision'].iloc[i - 1] == True:
                    col_num += 1
                    # if has_dis: col_dis.append(np.mean(dis_episode))
                    # if has_ttc: col_ttc.append(np.mean(ttc_episode))
                    # if has_comp:
                    #     col_comp.append(sum(comp_episode))
                    #     if 100 - sum(comp_episode) > 0.1:
                    #         col_comp_num += 1
                    #         if has_dis: col_comp_dis.append(np.mean(dis_episode))
                    #         if has_ttc: col_comp_ttc.append(np.mean(ttc_episode))
                    #         col_comp_comp.append(sum(comp_episode))
                    # if has_comf:
                    #     col_comf.append(np.mean(comf_episode))
                    #     if np.mean(comf_episode) != 0.0:
                    #         col_comf_num += 1
                    #         if has_dis: col_comf_dis.append(np.mean(dis_episode))
                    #         if has_ttc: col_comf_ttc.append(np.mean(ttc_episode))
                    #         col_comf_comf.append(np.mean(comf_episode))
                    # if has_spdif:
                    #     col_spdif.append(np.mean(spdif_episode))
                    #     if np.mean(spdif_episode) != 0.0:
                    #         col_spdif_num += 1
                    #         if has_dis: col_spdif_dis.append(np.mean(dis_episode))
                    #         if has_ttc: col_spdif_ttc.append(np.mean(ttc_episode))
                    #         col_spdif_spdif.append(np.mean(spdif_episode))
                    col_action_list.append(tuple(action))

                if has_dis and has_ttc:
                    if np.mean(dis_episode) < 5.0 and np.mean(ttc_episode) < 1.0:
                        dis_ttc_num += 1
                        dis_ttc_dis.append(np.mean(dis_episode))
                        dis_ttc_ttc.append(np.mean(ttc_episode))
                if has_dis and has_comp:
                    if np.mean(dis_episode) < 5.0 and 100 - sum(comp_episode) > 0.1:
                        dis_comp_num += 1
                        dis_comp_dis.append(np.mean(dis_episode))
                        dis_comp_comp.append(sum(comp_episode))
                if has_dis and has_comf:
                    if np.mean(dis_episode) < 5.0 and np.mean(comf_episode) > 0.9:
                        dis_comf_num += 1
                        dis_comf_dis.append(np.mean(dis_episode))
                        dis_comf_comf.append(np.mean(comf_episode))
                if has_dis and has_spdif:
                    if np.mean(dis_episode) < 5.0 and np.mean(spdif_episode) != 0.0:
                        dis_spdif_num += 1
                        dis_spdif_dis.append(np.mean(dis_episode))
                        dis_spdif_spdif.append(np.mean(spdif_episode))
                if has_ttc and has_comp:
                    if np.mean(ttc_episode) < 1.0 and 100 - sum(comp_episode) > 0.1:
                        ttc_comp_num += 1
                        ttc_comp_ttc.append(np.mean(ttc_episode))
                        ttc_comp_comp.append(sum(comp_episode))
                if has_ttc and has_comf:
                    if np.mean(ttc_episode) < 1.0 and np.mean(comf_episode) > 0.9:
                        ttc_comf_num += 1
                        ttc_comf_ttc.append(np.mean(ttc_episode))
                        ttc_comf_comf.append(np.mean(comf_episode))
                if has_ttc and has_spdif:
                    if np.mean(ttc_episode) < 1.0 and np.mean(spdif_episode) != 0.0:
                        ttc_spdif_num += 1
                        ttc_spdif_ttc.append(np.mean(ttc_episode))
                        ttc_spdif_spdif.append(np.mean(spdif_episode))
                if has_comp and has_comf:
                    if 100 - sum(comp_episode) > 0.1 and np.mean(comf_episode) > 0.9:
                        comp_comf_num += 1
                        comp_comf_comp.append(sum(comp_episode))
                        comp_comf_comf.append(np.mean(comf_episode))
                if has_comp and has_spdif:
                    if 100 - sum(comp_episode) > 0.1 and np.mean(spdif_episode) != 0.0:
                        comp_spdif_num += 1
                        comp_spdif_comp.append(sum(comp_episode))
                        comp_spdif_spdif.append(np.mean(spdif_episode))
                if has_comf and has_spdif:
                    if np.mean(comf_episode) > 0.9 and np.mean(spdif_episode) != 0.0:
                        comf_spdif_num += 1
                        comf_spdif_comf.append(np.mean(comf_episode))
                        comf_spdif_spdif.append(np.mean(spdif_episode))

            if has_dis: rew_dis.append(np.mean(reward_dis))
            if has_ttc: rew_ttc.append(np.mean(reward_ttc))
            if has_comp: rew_comp.append(np.mean(reward_comp))
            if has_comf: rew_comf.append(np.mean(reward_comf))
            if has_spdif: rew_spdif.append(np.mean(reward_spdif))
            rew_total.append(data['reward_total'].iloc[i - 1] / (data['step'].iloc[i - 1] + 1))
            collision.append(max(np.array(coll)))

            if i == len(data):
                break

            episode = data['episode'].iloc[i]
            loss_episode = []
            dis_episode, ttc_episode, comp_episode, comf_episode, spdif_episode = [], [], [], [], []
            reward_dis, reward_ttc, reward_comp, reward_comf, reward_spdif = [], [], [], [], []
            coll = []
            action = []

        if data['collision'].iloc[i] == True:
            col_data = 1
            dis_data = 0
            ttc_data = 0
        else:
            col_data = 0
            dis_data = data['distance'].iloc[i] if has_dis else 0
            ttc_data = min(data['time_to_collision'].iloc[i], 20) if has_ttc else 0

        if has_loss: loss_episode.append(data['loss'].iloc[i])
        if has_dis: dis_episode.append(dis_data); reward_dis.append(data['reward_distance'].iloc[i])
        if has_ttc: ttc_episode.append(ttc_data); reward_ttc.append(data['reward_time_to_collision'].iloc[i])
        if has_comp: comp_episode.append(data['completion'].iloc[i]); reward_comp.append(data['reward_completion'].iloc[i])
        if has_comf: comf_episode.append(data['comfort'].iloc[i]); reward_comf.append(data['reward_comfort'].iloc[i])
        if has_spdif: spdif_episode.append(data['speed_diff'].iloc[i]); reward_spdif.append(data['reward_speed_diff'].iloc[i])
        coll.append(col_data)
        action.append(data['action'].iloc[i])

    create_floder(f'{folder_path}/{file_name}')

    x = np.arange(episode_start, episode_end + 1, 1)

    def save_plot(d, title, fn):
        plt.figure(1)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Value')
        plt.plot(x, d, label='Original', color='#9abbda')
        plt.plot(x, tensorboard_smoothing(d), label='EWMA Smoothed', color='#4f70a4')
        plt.legend(loc='upper right')
        plt.savefig(f'{folder_path}/{file_name}/{fn}.png', format="png")
        plt.clf()

    if has_loss: save_plot(loss, 'Loss Convergence Trend', 'loss')
    if has_dis: save_plot(dis, 'Distance Convergence Trend', 'distance')
    if has_ttc: save_plot(ttc, 'Time to Collision Convergence Trend', 'time_to_collision')
    if has_comp: save_plot(comp, 'Route Completion Percentage Convergence Trend', 'completion')
    if has_comf: save_plot(comf, 'Jerk Convergence Trend', 'comfort')
    if has_spdif: save_plot(spdif, 'Speed Difference Convergence Trend', 'speed_diff')
    if has_dis: save_plot(rew_dis, 'Distance Reward Convergence Trend', 'reward_distance')
    if has_ttc: save_plot(rew_ttc, 'Time to Collision Reward Convergence Trend', 'reward_time_to_collision')
    if has_comp: save_plot(rew_comp, 'Route Completion Percentage Reward Convergence Trend', 'reward_completion')
    if has_comf: save_plot(rew_comf, 'Jerk Reward Convergence Trend', 'reward_comfort')
    if has_spdif: save_plot(rew_spdif, 'Speed Difference Reward Convergence Trend', 'reward_speed_diff')
    save_plot(rew_total, 'Total Reward Convergence Trend', 'reward_total')

    plt.figure(1)
    plt.title(f'Collision Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, collision, color='#4f70a4')
    plt.savefig(f'{folder_path}/{file_name}/collision.png', format="png")
    plt.clf()

    metrics_dict = {}
    statistic_metrics_dict = {}
    csv_metric_dict = {}

    with open(f'{folder_path}/{file_name}/collision.txt', 'w') as file:
        if has_dis:
            value = None if len(tol_dis) == 0 else sum(tol_dis) / len(tol_dis)
            file.write(f"{len(tol_dis)} scenario tol_dis average: {value}\n")
            metrics_dict['tol_dis'] = value
            statistic_metrics_dict['tol_dis'] = tol_dis
            csv_metric_dict['tol_x1'] = tol_dis
            csv_metric_dict['tol_x1_nor'] = normalize_list(tol_dis, 'distance')

            file.write(f"dis_num: {dis_num}\n")
            metrics_dict['dis_num'] = dis_num
            csv_metric_dict['x1_num'] = dis_num

            value = None if len(dis_num_dis) == 0 else sum(dis_num_dis) / len(dis_num_dis)
            file.write(f"{len(dis_num_dis)} scenario dis_num_dis average: {value}\n")
            metrics_dict['dis_num_dis'] = value
            statistic_metrics_dict['dis_num_dis'] = dis_num_dis
            csv_metric_dict['x1_num_x1'] = dis_num_dis
            csv_metric_dict['x1_num_x1_nor'] = normalize_list(dis_num_dis, 'distance')

        if has_ttc:
            value = None if len(tol_ttc) == 0 else sum(tol_ttc) / len(tol_ttc)
            file.write(f"{len(tol_ttc)} scenario tol_ttc average: {value}\n")
            metrics_dict['tol_ttc'] = value
            statistic_metrics_dict['tol_ttc'] = tol_ttc
            csv_metric_dict['tol_x1' if csv_metric_dict.get('tol_x1') is None else 'tol_x2'] = tol_ttc
            csv_metric_dict['tol_x1_nor' if csv_metric_dict.get('tol_x1_nor') is None else 'tol_x2_nor'] = \
                normalize_list(tol_ttc, 'time_to_collision')

            file.write(f"ttc_num: {ttc_num}\n")
            metrics_dict['ttc_num'] = ttc_num
            csv_metric_dict['x1_num' if csv_metric_dict.get('x1_num') is None else 'x2_num'] = ttc_num

            value = None if len(ttc_num_ttc) == 0 else sum(ttc_num_ttc) / len(ttc_num_ttc)
            file.write(f"{len(ttc_num_ttc)} scenario ttc_num_ttc average: {value}\n")
            metrics_dict['ttc_num_ttc'] = value
            statistic_metrics_dict['ttc_num_ttc'] = ttc_num_ttc
            csv_metric_dict['x1_num_x1' if csv_metric_dict.get('x1_num_x1') is None else 'x2_num_x2'] = ttc_num_ttc
            csv_metric_dict['x1_num_x1_nor' if csv_metric_dict.get('x1_num_x1_nor') is None else 'x2_num_x2_nor'] = \
                normalize_list(ttc_num_ttc, 'time_to_collision')

        if has_comp:
            value = None if len(tol_comp) == 0 else sum(tol_comp) / len(tol_comp)
            file.write(f"{len(tol_comp)} scenario tol_comp average: {value}\n")
            metrics_dict['tol_comp'] = value
            statistic_metrics_dict['tol_comp'] = tol_comp
            csv_metric_dict['tol_x1' if csv_metric_dict.get('tol_x1') is None else 'tol_x2'] = tol_comp
            csv_metric_dict['tol_x1_nor' if csv_metric_dict.get('tol_x1_nor') is None else 'tol_x2_nor'] = \
                normalize_list(tol_comp, 'completion')

            file.write(f"comp_num: {comp_num}\n")
            metrics_dict['comp_num'] = comp_num
            csv_metric_dict['x1_num' if csv_metric_dict.get('x1_num') is None else 'x2_num'] = comp_num

            value = None if len(comp_num_comp) == 0 else sum(comp_num_comp) / len(comp_num_comp)
            file.write(f"{len(comp_num_comp)} scenario comp_num_comp average: {value}\n")
            metrics_dict['comp_num_comp'] = value
            statistic_metrics_dict['comp_num_comp'] = comp_num_comp
            csv_metric_dict['x1_num_x1' if csv_metric_dict.get('x1_num_x1') is None else 'x2_num_x2'] = comp_num_comp
            csv_metric_dict['x1_num_x1_nor' if csv_metric_dict.get('x1_num_x1_nor') is None else 'x2_num_x2_nor'] = \
                normalize_list(comp_num_comp, 'completion')

        if has_comf:
            value = None if len(tol_comf) == 0 else sum(tol_comf) / len(tol_comf)
            file.write(f"{len(tol_comf)} scenario tol_comf average: {value}\n")
            metrics_dict['tol_comf'] = value
            statistic_metrics_dict['tol_comf'] = tol_comf
            csv_metric_dict['tol_x1' if csv_metric_dict.get('tol_x1') is None else 'tol_x2'] = tol_comf
            csv_metric_dict['tol_x1_nor' if csv_metric_dict.get('tol_x1_nor') is None else 'tol_x2_nor'] = \
                normalize_list(tol_comf, 'comfort')

            file.write(f"comf_num: {comf_num}\n")
            metrics_dict['comf_num'] = comf_num
            csv_metric_dict['x1_num' if csv_metric_dict.get('x1_num') is None else 'x2_num'] = comf_num

            value = None if len(comf_num_comf) == 0 else sum(comf_num_comf) / len(comf_num_comf)
            file.write(f"{len(comf_num_comf)} scenario comf_num_comf average: {value}\n")
            metrics_dict['comf_num_comf'] = value
            statistic_metrics_dict['comf_num_comf'] = comf_num_comf
            csv_metric_dict['x1_num_x1' if csv_metric_dict.get('x1_num_x1') is None else 'x2_num_x2'] = comf_num_comf
            csv_metric_dict['x1_num_x1_nor' if csv_metric_dict.get('x1_num_x1_nor') is None else 'x2_num_x2_nor'] = \
                normalize_list(comf_num_comf, 'comfort')

        if has_spdif:
            value = None if len(tol_spdif) == 0 else sum(tol_spdif) / len(tol_spdif)
            file.write(f"{len(tol_spdif)} scenario tol_spdif average: {value}\n")
            metrics_dict['tol_spdif'] = value
            statistic_metrics_dict['tol_spdif'] = tol_spdif
            csv_metric_dict['tol_x1' if csv_metric_dict.get('tol_x1') is None else 'tol_x2'] = tol_spdif
            csv_metric_dict['tol_x1_nor' if csv_metric_dict.get('tol_x1_nor') is None else 'tol_x2_nor'] = \
                normalize_list(tol_spdif, 'speed_diff')

            file.write(f"spdif_num: {spdif_num}\n")
            metrics_dict['spdif_num'] = spdif_num
            csv_metric_dict['x1_num' if csv_metric_dict.get('x1_num') is None else 'x2_num'] = spdif_num

            value = None if len(spdif_num_spdif) == 0 else sum(spdif_num_spdif) / len(spdif_num_spdif)
            file.write(f"{len(spdif_num_spdif)} scenario spdif_num_spdif average: {value}\n")
            metrics_dict['spdif_num_spdif'] = value
            statistic_metrics_dict['spdif_num_spdif'] = spdif_num_spdif
            csv_metric_dict['x1_num_x1' if csv_metric_dict.get('x1_num_x1') is None else 'x2_num_x2'] = spdif_num_spdif
            csv_metric_dict['x1_num_x1_nor' if csv_metric_dict.get('x1_num_x1_nor') is None else 'x2_num_x2_nor'] = \
                normalize_list(spdif_num_spdif, 'speed_diff')

        file.write(f"col_num: {col_num}\n")
        metrics_dict['col_num'] = col_num
        csv_metric_dict['col_num'] = col_num

        # if has_dis:
        #     value = None if len(col_dis) == 0 else sum(col_dis) / len(col_dis)
        #     file.write(f"{len(col_dis)} scenario col_dis average: {value}\n")
        #     metrics_dict['col_dis'] = value
        #     statistic_metrics_dict['col_dis'] = col_dis

        # if has_ttc:
        #     value = None if len(col_ttc) == 0 else sum(col_ttc) / len(col_ttc)
        #     file.write(f"{len(col_ttc)} scenario col_ttc average: {value}\n")
        #     metrics_dict['col_ttc'] = value
        #     statistic_metrics_dict['col_ttc'] = col_ttc

        # if has_comp:
        #     value = None if len(col_comp) == 0 else sum(col_comp) / len(col_comp)
        #     file.write(f"{len(col_comp)} scenario col_comp average: {value}\n")
        #     metrics_dict['col_comp'] = value
        #     statistic_metrics_dict['col_comp'] = col_comp

        # if has_comf:
        #     value = None if len(col_comf) == 0 else sum(col_comf) / len(col_comf)
        #     file.write(f"{len(col_comf)} scenario col_comf average: {value}\n")
        #     metrics_dict['col_comf'] = value
        #     statistic_metrics_dict['col_comf'] = col_comf

        # if has_spdif:
        #     value = None if len(col_spdif) == 0 else sum(col_spdif) / len(col_spdif)
        #     file.write(f"{len(col_spdif)} scenario col_spdif average: {value}\n")
        #     metrics_dict['col_spdif'] = value
        #     statistic_metrics_dict['col_spdif'] = col_spdif

        # if has_dis and has_comp:
        #     file.write(f"col_comp_num: {col_comp_num}\n")
        #     metrics_dict['col_comp_num'] = col_comp_num
        #
        #     value = None if len(col_comp_dis) == 0 else sum(col_comp_dis) / len(col_comp_dis)
        #     file.write(f"{len(col_comp_dis)} scenario col_comp_dis average: {value}\n")
        #     metrics_dict['col_comp_dis'] = value
        #     statistic_metrics_dict['col_comp_dis'] = col_comp_dis
        #
        #     value = None if len(col_comp_comp) == 0 else sum(col_comp_comp) / len(col_comp_comp)
        #     file.write(f"{len(col_comp_comp)} scenario col_comp_comp average: {value}\n")
        #     metrics_dict['col_comp_comp'] = value
        #     statistic_metrics_dict['col_comp_comp'] = col_comp_comp

        # if has_ttc and has_comp:
        #     file.write(f"col_comp_num: {col_comp_num}\n")
        #     metrics_dict['col_comp_num'] = col_comp_num
        #
        #     value = None if len(col_comp_ttc) == 0 else sum(col_comp_ttc) / len(col_comp_ttc)
        #     file.write(f"{len(col_comp_ttc)} scenario col_comp_ttc average: {value}\n")
        #     metrics_dict['col_comp_ttc'] = value
        #     statistic_metrics_dict['col_comp_ttc'] = col_comp_ttc
        #
        #     value = None if len(col_comp_comp) == 0 else sum(col_comp_comp) / len(col_comp_comp)
        #     file.write(f"{len(col_comp_comp)} scenario col_comp_comp average: {value}\n")
        #     metrics_dict['col_comp_comp'] = value
        #     statistic_metrics_dict['col_comp_comp'] = col_comp_comp

        # if has_dis and has_comf:
        #     file.write(f"col_comf_num: {col_comf_num}\n")
        #     metrics_dict['col_comf_num'] = col_comf_num
        #
        #     value = None if len(col_comf_dis) == 0 else sum(col_comf_dis) / len(col_comf_dis)
        #     file.write(f"{len(col_comf_dis)} scenario col_comf_dis average: {value}\n")
        #     metrics_dict['col_comf_dis'] = value
        #     statistic_metrics_dict['col_comf_dis'] = col_comf_dis
        #
        #     value = None if len(col_comf_comf) == 0 else sum(col_comf_comf) / len(col_comf_comf)
        #     file.write(f"{len(col_comf_comf)} scenario col_comf_comf average: {value}\n")
        #     metrics_dict['col_comf_comf'] = value
        #     statistic_metrics_dict['col_comf_comf'] = col_comf_comf

        # if has_ttc and has_comf:
        #     file.write(f"col_comf_num: {col_comf_num}\n")
        #     metrics_dict['col_comf_num'] = col_comf_num
        #
        #     value = None if len(col_comf_ttc) == 0 else sum(col_comf_ttc) / len(col_comf_ttc)
        #     file.write(f"{len(col_comf_ttc)} scenario col_comf_ttc average: {value}\n")
        #     metrics_dict['col_comf_ttc'] = value
        #     statistic_metrics_dict['col_comf_ttc'] = col_comf_ttc
        #
        #     value = None if len(col_comf_comf) == 0 else sum(col_comf_comf) / len(col_comf_comf)
        #     file.write(f"{len(col_comf_comf)} scenario col_comf_comf average: {value}\n")
        #     metrics_dict['col_comf_comf'] = value
        #     statistic_metrics_dict['col_comf_comf'] = col_comf_comf

        # if has_dis and has_spdif:
        #     file.write(f"col_spdif_num: {col_spdif_num}\n")
        #     metrics_dict['col_spdif_num'] = col_spdif_num
        #
        #     value = None if len(col_spdif_dis) == 0 else sum(col_spdif_dis) / len(col_spdif_dis)
        #     file.write(f"{len(col_spdif_dis)} scenario col_spdif_dis average: {value}\n")
        #     metrics_dict['col_spdif_dis'] = value
        #     statistic_metrics_dict['col_spdif_dis'] = col_spdif_dis
        #
        #     value = None if len(col_spdif_spdif) == 0 else sum(col_spdif_spdif) / len(col_spdif_spdif)
        #     file.write(f"{len(col_spdif_spdif)} scenario col_spdif_spdif average: {value}\n")
        #     metrics_dict['col_spdif_spdif'] = value
        #     statistic_metrics_dict['col_spdif_spdif'] = col_spdif_spdif

        # if has_ttc and has_spdif:
        #     file.write(f"col_spdif_num: {col_spdif_num}\n")
        #     metrics_dict['col_spdif_num'] = col_spdif_num
        #
        #     value = None if len(col_spdif_ttc) == 0 else sum(col_spdif_ttc) / len(col_spdif_ttc)
        #     file.write(f"{len(col_spdif_ttc)} scenario col_spdif_ttc average: {value}\n")
        #     metrics_dict['col_spdif_ttc'] = value
        #     statistic_metrics_dict['col_spdif_ttc'] = col_spdif_ttc
        #
        #     value = None if len(col_spdif_spdif) == 0 else sum(col_spdif_spdif) / len(col_spdif_spdif)
        #     file.write(f"{len(col_spdif_spdif)} scenario col_spdif_spdif average: {value}\n")
        #     metrics_dict['col_spdif_spdif'] = value
        #     statistic_metrics_dict['col_spdif_spdif'] = col_spdif_spdif

        if has_dis and has_ttc:
            file.write(f"dis_ttc_num: {dis_ttc_num}\n")
            metrics_dict['dis_ttc_num'] = dis_ttc_num
            csv_metric_dict['x1_x2_num'] = dis_ttc_num

            value = None if len(dis_ttc_dis) == 0 else sum(dis_ttc_dis) / len(dis_ttc_dis)
            file.write(f"{len(dis_ttc_dis)} scenario dis_ttc_dis average: {value}\n")
            metrics_dict['dis_ttc_dis'] = value
            statistic_metrics_dict['dis_ttc_dis'] = dis_ttc_dis
            csv_metric_dict['x1_x2_x1'] = dis_ttc_dis
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(dis_ttc_dis, 'distance')

            value = None if len(dis_ttc_ttc) == 0 else sum(dis_ttc_ttc) / len(dis_ttc_ttc)
            file.write(f"{len(dis_ttc_ttc)} scenario dis_ttc_ttc average: {value}\n")
            metrics_dict['dis_ttc_ttc'] = value
            statistic_metrics_dict['dis_ttc_ttc'] = dis_ttc_ttc
            csv_metric_dict['x1_x2_x2'] = dis_ttc_ttc
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(dis_ttc_ttc, 'time_to_collision')

        if has_dis and has_comp:
            file.write(f"dis_comp_num: {dis_comp_num}\n")
            metrics_dict['dis_comp_num'] = dis_comp_num
            csv_metric_dict['x1_x2_num'] = dis_comp_num

            value = None if len(dis_comp_dis) == 0 else sum(dis_comp_dis) / len(dis_comp_dis)
            file.write(f"{len(dis_comp_dis)} scenario dis_comp_dis average: {value}\n")
            metrics_dict['dis_comp_dis'] = value
            statistic_metrics_dict['dis_comp_dis'] = dis_comp_dis
            csv_metric_dict['x1_x2_x1'] = dis_comp_dis
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(dis_comp_dis, 'distance')

            value = None if len(dis_comp_comp) == 0 else sum(dis_comp_comp) / len(dis_comp_comp)
            file.write(f"{len(dis_comp_comp)} scenario dis_comp_comp average: {value}\n")
            metrics_dict['dis_comp_comp'] = value
            statistic_metrics_dict['dis_comp_comp'] = dis_comp_comp
            csv_metric_dict['x1_x2_x2'] = dis_comp_comp
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(dis_comp_comp, 'completion')

        if has_dis and has_comf:
            file.write(f"dis_comf_num: {dis_comf_num}\n")
            metrics_dict['dis_comf_num'] = dis_comf_num
            csv_metric_dict['x1_x2_num'] = dis_comf_num

            value = None if len(dis_comf_dis) == 0 else sum(dis_comf_dis) / len(dis_comf_dis)
            file.write(f"{len(dis_comf_dis)} scenario dis_comf_dis average: {value}\n")
            metrics_dict['dis_comf_dis'] = value
            statistic_metrics_dict['dis_comf_dis'] = dis_comf_dis
            csv_metric_dict['x1_x2_x1'] = dis_comf_dis
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(dis_comf_dis, 'distance')

            value = None if len(dis_comf_comf) == 0 else sum(dis_comf_comf) / len(dis_comf_comf)
            file.write(f"{len(dis_comf_comf)} scenario dis_comf_comf average: {value}\n")
            metrics_dict['dis_comf_comf'] = value
            statistic_metrics_dict['dis_comf_comf'] = dis_comf_comf
            csv_metric_dict['x1_x2_x2'] = dis_comf_comf
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(dis_comf_comf, 'comfort')

        if has_dis and has_spdif:
            file.write(f"dis_spdif_num: {dis_spdif_num}\n")
            metrics_dict['dis_spdif_num'] = dis_spdif_num
            csv_metric_dict['x1_x2_num'] = dis_spdif_num

            value = None if len(dis_spdif_dis) == 0 else sum(dis_spdif_dis) / len(dis_spdif_dis)
            file.write(f"{len(dis_spdif_dis)} scenario dis_spdif_dis average: {value}\n")
            metrics_dict['dis_spdif_dis'] = value
            statistic_metrics_dict['dis_spdif_dis'] = dis_spdif_dis
            csv_metric_dict['x1_x2_x1'] = dis_spdif_dis
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(dis_spdif_dis, 'distance')

            value = None if len(dis_spdif_spdif) == 0 else sum(dis_spdif_spdif) / len(dis_spdif_spdif)
            file.write(f"{len(dis_spdif_spdif)} scenario dis_spdif_spdif average: {value}\n")
            metrics_dict['dis_spdif_spdif'] = value
            statistic_metrics_dict['dis_spdif_spdif'] = dis_spdif_spdif
            csv_metric_dict['x1_x2_x2'] = dis_spdif_spdif
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(dis_spdif_spdif, 'speed_diff')

        if has_ttc and has_comp:
            file.write(f"ttc_comp_num: {ttc_comp_num}\n")
            metrics_dict['ttc_comp_num'] = ttc_comp_num
            csv_metric_dict['x1_x2_num'] = ttc_comp_num

            value = None if len(ttc_comp_ttc) == 0 else sum(ttc_comp_ttc) / len(ttc_comp_ttc)
            file.write(f"{len(ttc_comp_ttc)} scenario ttc_comp_ttc average: {value}\n")
            metrics_dict['ttc_comp_ttc'] = value
            statistic_metrics_dict['ttc_comp_ttc'] = ttc_comp_ttc
            csv_metric_dict['x1_x2_x1'] = ttc_comp_ttc
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(ttc_comp_ttc, 'time_to_collision')

            value = None if len(ttc_comp_comp) == 0 else sum(ttc_comp_comp) / len(ttc_comp_comp)
            file.write(f"{len(ttc_comp_comp)} scenario ttc_comp_comp average: {value}\n")
            metrics_dict['ttc_comp_comp'] = value
            statistic_metrics_dict['ttc_comp_comp'] = ttc_comp_comp
            csv_metric_dict['x1_x2_x2'] = ttc_comp_comp
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(ttc_comp_comp, 'completion')

        if has_ttc and has_comf:
            file.write(f"ttc_comf_num: {ttc_comf_num}\n")
            metrics_dict['ttc_comf_num'] = ttc_comf_num
            csv_metric_dict['x1_x2_num'] = ttc_comf_num

            value = None if len(ttc_comf_ttc) == 0 else sum(ttc_comf_ttc) / len(ttc_comf_ttc)
            file.write(f"{len(ttc_comf_ttc)} scenario ttc_comf_ttc average: {value}\n")
            metrics_dict['ttc_comf_ttc'] = value
            statistic_metrics_dict['ttc_comf_ttc'] = ttc_comf_ttc
            csv_metric_dict['x1_x2_x1'] = ttc_comf_ttc
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(ttc_comf_ttc, 'time_to_collision')

            value = None if len(ttc_comf_comf) == 0 else sum(ttc_comf_comf) / len(ttc_comf_comf)
            file.write(f"{len(ttc_comf_comf)} scenario ttc_comf_comf average: {value}\n")
            metrics_dict['ttc_comf_comf'] = value
            statistic_metrics_dict['ttc_comf_comf'] = ttc_comf_comf
            csv_metric_dict['x1_x2_x2'] = ttc_comf_comf
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(ttc_comf_comf, 'comfort')

        if has_ttc and has_spdif:
            file.write(f"ttc_spdif_num: {ttc_spdif_num}\n")
            metrics_dict['ttc_spdif_num'] = ttc_spdif_num
            csv_metric_dict['x1_x2_num'] = ttc_spdif_num

            value = None if len(ttc_spdif_ttc) == 0 else sum(ttc_spdif_ttc) / len(ttc_spdif_ttc)
            file.write(f"{len(ttc_spdif_ttc)} scenario ttc_spdif_ttc average: {value}\n")
            metrics_dict['ttc_spdif_ttc'] = value
            statistic_metrics_dict['ttc_spdif_ttc'] = ttc_spdif_ttc
            csv_metric_dict['x1_x2_x1'] = ttc_spdif_ttc
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(ttc_spdif_ttc, 'time_to_collision')

            value = None if len(ttc_spdif_spdif) == 0 else sum(ttc_spdif_spdif) / len(ttc_spdif_spdif)
            file.write(f"{len(ttc_spdif_spdif)} scenario ttc_spdif_spdif average: {value}\n")
            metrics_dict['ttc_spdif_spdif'] = value
            statistic_metrics_dict['ttc_spdif_spdif'] = ttc_spdif_spdif
            csv_metric_dict['x1_x2_x2'] = ttc_spdif_spdif
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(ttc_spdif_spdif, 'speed_diff')

        if has_comp and has_comf:
            file.write(f"comp_comf_num: {comp_comf_num}\n")
            metrics_dict['comp_comf_num'] = comp_comf_num
            csv_metric_dict['x1_x2_num'] = comp_comf_num

            value = None if len(comp_comf_comp) == 0 else sum(comp_comf_comp) / len(comp_comf_comp)
            file.write(f"{len(comp_comf_comp)} scenario comp_comf_comp average: {value}\n")
            metrics_dict['comp_comf_comp'] = value
            statistic_metrics_dict['comp_comf_comp'] = comp_comf_comp
            csv_metric_dict['x1_x2_x1'] = comp_comf_comp
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(comp_comf_comp, 'completion')

            value = None if len(comp_comf_comf) == 0 else sum(comp_comf_comf) / len(comp_comf_comf)
            file.write(f"{len(comp_comf_comf)} scenario comp_comf_comf average: {value}\n")
            metrics_dict['comp_comf_comf'] = value
            statistic_metrics_dict['comp_comf_comf'] = comp_comf_comf
            csv_metric_dict['x1_x2_x2'] = comp_comf_comf
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(comp_comf_comf, 'comfort')

        if has_comp and has_spdif:
            file.write(f"comp_spdif_num: {comp_spdif_num}\n")
            metrics_dict['comp_spdif_num'] = comp_spdif_num
            csv_metric_dict['x1_x2_num'] = comp_spdif_num

            value = None if len(comp_spdif_comp) == 0 else sum(comp_spdif_comp) / len(comp_spdif_comp)
            file.write(f"{len(comp_spdif_comp)} scenario comp_spdif_comp average: {value}\n")
            metrics_dict['comp_spdif_comp'] = value
            statistic_metrics_dict['comp_spdif_comp'] = comp_spdif_comp
            csv_metric_dict['x1_x2_x1'] = comp_spdif_comp
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(comp_spdif_comp, 'completion')

            value = None if len(comp_spdif_spdif) == 0 else sum(comp_spdif_spdif) / len(comp_spdif_spdif)
            file.write(f"{len(comp_spdif_spdif)} scenario comp_spdif_spdif average: {value}\n")
            metrics_dict['comp_spdif_spdif'] = value
            statistic_metrics_dict['comp_spdif_spdif'] = comp_spdif_spdif
            csv_metric_dict['x1_x2_x2'] = comp_spdif_spdif
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(comp_spdif_spdif, 'speed_diff')

        if has_comf and has_spdif:
            file.write(f"comf_spdif_num: {comf_spdif_num}\n")
            metrics_dict['comf_spdif_num'] = comf_spdif_num
            csv_metric_dict['x1_x2_num'] = comf_spdif_num

            value = None if len(comf_spdif_comf) == 0 else sum(comf_spdif_comf) / len(comf_spdif_comf)
            file.write(f"{len(comf_spdif_comf)} scenario comf_spdif_comf average: {value}\n")
            metrics_dict['comf_spdif_comf'] = value
            statistic_metrics_dict['comf_spdif_comf'] = comf_spdif_comf
            csv_metric_dict['x1_x2_x1'] = comf_spdif_comf
            csv_metric_dict['x1_x2_x1_nor'] = normalize_list(comf_spdif_comf, 'comfort')

            value = None if len(comf_spdif_spdif) == 0 else sum(comf_spdif_spdif) / len(comf_spdif_spdif)
            file.write(f"{len(comf_spdif_spdif)} scenario comf_spdif_spdif average: {value}\n")
            metrics_dict['comf_spdif_spdif'] = value
            statistic_metrics_dict['comf_spdif_spdif'] = comf_spdif_spdif
            csv_metric_dict['x1_x2_x2'] = comf_spdif_spdif
            csv_metric_dict['x1_x2_x2_nor'] = normalize_list(comf_spdif_spdif, 'speed_diff')

        counter = Counter(action_list)
        file.write(f"\n{len(counter)} sets of actions for unique scenarios")
        metrics_dict['unique_action_set_num'] = len(counter)
        csv_metric_dict['unique_action_set_num'] = len(counter)
        for k, v in counter.items():
            file.write(f"\naction: {k} number: {v}")
        action_diff, action_diff_no_num = compute_action_difference(action_list)
        file.write(f"\n\naverage action difference (with scenario number): {action_diff}")
        file.write(f"\naverage action difference (without scenario number): {action_diff_no_num}\n")
        metrics_dict['unique_action_diff'] = action_diff
        metrics_dict['unique_action_diff_num_no'] = action_diff_no_num
        csv_metric_dict['unique_action_diff'] = action_diff
        csv_metric_dict['unique_action_diff_num_no'] = action_diff_no_num

        # counter = Counter(col_action_list)
        # file.write(f"\n{len(counter)} sets of actions that cause unique collisions")
        # metrics_dict['collision_action_set'] = len(counter)
        # for k, v in counter.items():
        #     file.write(f"\naction: {k} number: {v}")
        # action_diff, action_diff_no_num = compute_action_difference(col_action_list)
        # file.write(f"\n\naverage action difference (with scenario number): {action_diff}")
        # file.write(f"\naverage action difference (without scenario number): {action_diff_no_num}\n")
        # metrics_dict['collision_action_diff'] = action_diff
        # metrics_dict['collision_action_diff_no_num'] = action_diff_no_num

        # all_diff, collision_diff = json_compute_difference(folder_path, file_name)
        # file.write(f"\naverage all scenario distance difference: {all_diff}")
        # file.write(f"\naverage collision scenario distance difference: {collision_diff}")
        # metrics_dict['all_scenario_diff'] = all_diff
        # metrics_dict['collision_scenario_diff'] = collision_diff

        # all_diff, collision_diff = json_compute_differences(folder_path, file_name)
        # for k, v in all_diff.items():
        #     file.write(f"\naverage all scenario {k} difference: {v}")
        #     metrics_dict[f'all_scenario_diff_{k}'] = v
        # for k, v in collision_diff.items():
        #     file.write(f"\naverage collision scenario {k} difference: {v}")
        #     metrics_dict[f'collision_scenario_diff_{k}'] = v

        all_dtw, collision_dtw = json_compute_dtw_pairwise(folder_path, file_name)
        file.write(f"\nall scenario dynamic_time_warping difference: {all_dtw}")
        # file.write(f"\ncollision scenario dynamic_time_warping difference: {collision_dtw}")
        metrics_dict['all_scenario_diff'] = all_dtw
        csv_metric_dict['all_scenario_diff'] = all_dtw
        # metrics_dict['collision_scenario_diff'] = collision_dtw

    # data_to_save = {'episode': x}
    # fn = '3D'
    # if has_dis: data_to_save['dis'] = dis; fn += '_dis'
    # if has_ttc: data_to_save['ttc'] = ttc; fn += '_ttc'
    # if has_comp: data_to_save['comp'] = comp; fn += '_comp'
    # if has_comf: data_to_save['comf'] = comf; fn += '_comf'
    # if has_spdif: data_to_save['spdif'] = spdif; fn += '_spdif'
    # with open(f'{folder_path}/{file_name}/{fn}.pkl', 'wb') as f:
    #     pickle.dump(data_to_save, f)

    custom_row = ["tol_x1", "tol_x1_nor", "tol_x2", "tol_x2_nor", "x1_num", "x1_num_x1", "x1_num_x1_nor", "x2_num",
                  "x2_num_x2", "x2_num_x2_nor", "x1_x2_num", "x1_x2_x1", "x1_x2_x1_nor", "x1_x2_x2", "x1_x2_x2_nor",
                  "col_num", "unique_action_set_num", "unique_action_diff", "unique_action_diff_num_no", "all_scenario_diff"]
    save_dir = os.path.join(folder_path, file_name)
    csv_path = os.path.join(save_dir, 'metrics.csv')
    pkl_path = os.path.join(save_dir, 'metrics.pkl')
    normalized = {}
    for col in custom_row:
        val = csv_metric_dict.get(col, None)
        if isinstance(val, list):
            normalized[col] = val
        else:
            normalized[col] = [val]
    max_len = max(len(v) for v in normalized.values())
    table = {'index': list(range(1, max_len + 1))}
    for col in custom_row:
        values = normalized[col]
        table[col] = [
            values[i] if i < len(values) else None
            for i in range(max_len)
        ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index'] + custom_row)
        for i in range(max_len):
            row = [table['index'][i]] + [table[col][i] for col in custom_row]
            writer.writerow(row)
    with open(pkl_path, 'wb') as f:
        pickle.dump(table, f)

    return metrics_dict, statistic_metrics_dict


def generate_metric_csv(root_dir, algorithm_list, objective_list, scenario_list):
    for i in range(len(objective_list) - 1):
        obj1 = objective_list[i]
        for j in range(i + 1, len(objective_list)):
            obj2 = objective_list[j]
            for scenario in scenario_list:
                for algorithm in algorithm_list:
                    if algorithm == 'random':
                        folder_path = f"{root_dir}/{algorithm}"
                    else:
                        folder_path = f"{root_dir}/{algorithm}/eval/{obj1}_{obj2}"
                    file_name = next(
                        file for file in os.listdir(folder_path)
                        if file.endswith('.csv') and scenario in file
                    )
                    f_name = os.path.splitext(file_name)[0]
                    full_path = os.path.join(folder_path, file_name)
                    data = pd.read_csv(full_path)
                    converge(data, f_name, folder_path)


def significance_label(p):
    if p is None:
        return 'NA'
    if p < 0.01:
        return '<0.01'
    elif p < 0.05:
        return '<0.05'
    else:
        return '>=0.05'


def a12_effect_label(a12):
    if a12 is None:
        return 'NA'
    a12 = max(a12, 1.0 - a12)
    if a12 < 0.556:
        return 'negligible'
    elif a12 < 0.638:
        return 'small'
    elif a12 < 0.714:
        return 'medium'
    else:
        return 'large'


def draw_group_brace(ax, target_texts, y_offset, label):
    fig = plt.gcf()
    xs = []
    for tick in ax.get_xticklabels():
        text = tick.get_text()
        if text in target_texts:
            x = tick.get_position()[0]
            trans = ax.get_xaxis_transform()
            inv = fig.transFigure.inverted()
            x_fig, y_fig = inv.transform(trans.transform((x, 0)))
            xs.append(x_fig)
    if not xs:
        return

    y = y_fig - y_offset

    if len(xs) == 1:
        x = xs[0]
        fig.lines.append(
            plt.Line2D(
                [x, x],
                [y + 0.02, y - 0.01],
                transform=fig.transFigure,
                color='black',
                linewidth=2
            )
        )
        txt = fig.text(
            x, y - 0.02,
            label,
            ha='center',
            va='top',
            fontsize=20,
            fontweight='bold'
        )
        txt.set_path_effects([
            path_effects.withStroke(linewidth=0.5, foreground='black'),
            path_effects.Normal()
        ])
        return

    x_min, x_max = min(xs), max(xs)
    x_mid = (x_min + x_max) / 2
    renderer = fig.canvas.get_renderer()
    tmp_text = fig.text(0, 0, label, fontsize=20, fontweight='bold')
    bbox = tmp_text.get_window_extent(renderer=renderer)
    tmp_text.remove()
    fig_width_px = fig.get_size_inches()[0] * fig.dpi
    text_width = bbox.width / fig_width_px
    gap = text_width / 2 + 0.005
    fig.lines.append(
        plt.Line2D(
            [x_min, x_mid - gap],
            [y - 0.015, y - 0.015],
            transform=fig.transFigure,
            color='black',
            linewidth=2
        )
    )
    fig.lines.append(
        plt.Line2D(
            [x_mid + gap, x_max],
            [y - 0.015, y - 0.015],
            transform=fig.transFigure,
            color='black',
            linewidth=2
        )
    )
    fig.lines.append(
        plt.Line2D(
            [x_min, x_min],
            [y - 0.015, y + 0.02],
            transform=fig.transFigure,
            color='black',
            linewidth=2
        )
    )
    fig.lines.append(
        plt.Line2D(
            [x_max, x_max],
            [y - 0.015, y + 0.02],
            transform=fig.transFigure,
            color='black',
            linewidth=2
        )
    )
    txt = fig.text(
        x_mid, y - 0.015,
        label,
        ha='center',
        va='center',
        fontsize=20,
        fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5)
    )
    txt.set_path_effects([
        path_effects.withStroke(linewidth=0.5, foreground='black'),
        path_effects.Normal()
    ])


def RQ1_box_plot_statistic(root_dir, algorithm_list, objective_list, scenario_list):
    plot_metrics = {
        'tol_x_nor': ['tol_x1_nor', 'tol_x2_nor'],
        'x_num': ['x1_num', 'x2_num'],
        'x_num_x_nor': ['x1_num_x1_nor', 'x2_num_x2_nor'],
        'x1_x2_num': ['x1_x2_num'],
        'x_x_x_nor': ['x1_x2_x1_nor', 'x1_x2_x2_nor'],
        'col_num': ['col_num'],
        'unique_action_set_num': ['unique_action_set_num'],
        'unique_action_diff_num_no': ['unique_action_diff_num_no'],
        'unique_action_diff': ['unique_action_diff'],
        'all_scenario_diff': ['all_scenario_diff']
    }
    metrics_process = {
        'x_num': lambda x: x / 100,
        'x1_x2_num': lambda x: x / 100,
        'col_num': lambda x: x / 100,
        'unique_action_set_num': lambda x: x / 100,
        'unique_action_diff_num_no': lambda x: x / 5,
        'unique_action_diff': lambda x: x / 5,
        'all_scenario_diff': lambda x: x / (x + 1),
    }
    plot_data = []
    raw_stats_data = {}
    for algorithm in algorithm_list:
        temp_metric_values = {m: [] for m in plot_metrics.keys()}
        for i in range(len(objective_list) - 1):
            obj1 = objective_list[i]
            for j in range(i + 1, len(objective_list)):
                obj2 = objective_list[j]
                for scenario in scenario_list:
                    if algorithm == 'random':
                        folder_path = f"{root_dir}/{algorithm}"
                    else:
                        folder_path = f"{root_dir}/{algorithm}/eval/{obj1}_{obj2}"
                    f_path = next(
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if os.path.isdir(os.path.join(folder_path, f)) and scenario in f
                    )
                    pkl_path = os.path.join(f_path, 'metrics.pkl')
                    with open(pkl_path, 'rb') as f:
                        data_dict = pickle.load(f)
                    for metric_name, col_list in plot_metrics.items():
                        for col in col_list:
                            if col in data_dict:
                                temp_metric_values[metric_name].extend(data_dict[col])
        for metric_name, all_values in temp_metric_values.items():
            raw_values = np.array(all_values, dtype=float)
            valid_mask = np.isfinite(raw_values)
            raw_values_clean = raw_values[valid_mask]
            if metric_name not in raw_stats_data:
                raw_stats_data[metric_name] = {}
            raw_stats_data[metric_name][algorithm] = raw_values_clean.copy()
            plot_values = raw_values_clean.copy()
            if metric_name in metrics_process:
                plot_values = metrics_process[metric_name](plot_values)
            for v in plot_values:
                plot_data.append({
                    'metric': metric_name,
                    'algorithm': algorithm,
                    'value': v
                })

    metric_display_name = {
        'tol_x_nor': r'$\mathit{OV}$',
        'x_num': r'$\#\mathit{SV}$',
        'x_num_x_nor': r'$\mathit{SVS}$',
        'x1_x2_num': r'$\#\mathit{MV}$',
        'x_x_x_nor': r'$\mathit{MVS}$',
        'col_num': r'$\#\mathit{C}$',
        'unique_action_set_num': r'$\#\mathit{UB}$',
        'unique_action_diff_num_no': r'$\mathit{UBD}$',
        'unique_action_diff': r'$\mathit{WBD}$',
        'all_scenario_diff': r'$\mathit{SCD}$',
    }
    plot_df = pd.DataFrame(plot_data)
    sns.set_theme(style="whitegrid", font_scale=2.2)
    plt.figure(figsize=(25, 6))
    color_palette = ["#4E79A7", "#E15759"]
    ax = sns.violinplot(
        data=plot_df,
        x='metric',
        y='value',
        hue='algorithm',
        density_norm='width',
        width=0.8,
        cut=0,
        inner="box",
        palette=color_palette,
        saturation=0.8,
        linewidth=2.5,
        zorder=1,
        gap=0.07
    )
    for line in ax.lines:
        orig_lw = line.get_linewidth()
        line.set_linewidth(orig_lw * 0.7)
    # for art in ax.collections:
    #     fc = art.get_facecolor()
    #     if fc.size > 0:
    #         if np.allclose(fc[0, :3], [1.0, 1.0, 1.0]):
    #             art.set_visible(False)
    for line in ax.lines:
        ydata = line.get_ydata()
        if np.allclose(ydata[0], ydata[-1]):
            line.set_visible(False)
    sns.boxplot(
        data=plot_df,
        x='metric',
        y='value',
        hue='algorithm',
        dodge=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "7",
            "zorder": 5
        },
        showbox=False,
        showcaps=False,
        showfliers=False,
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        palette=color_palette,
        zorder=2
    )
    mean_df = plot_df.groupby(['metric', 'algorithm'])['value'].mean().reset_index()
    metrics = list(plot_df['metric'].unique())
    algorithms = list(plot_df['algorithm'].unique())
    offset = 0.2
    for _, row in mean_df.iterrows():
        metric_idx = metrics.index(row['metric'])
        algo_idx = algorithms.index(row['algorithm'])
        x = metric_idx + (-offset if algo_idx == 0 else offset)
        y = row['value']
        ax.text(
            x,
            y + 0.02,
            f"{y:.3f}",
            ha='center',
            va='bottom',
            fontsize=18,
            fontweight='bold',
            color='white',
            zorder=6,
            path_effects=[
                path_effects.withStroke(linewidth=4, foreground="black")
            ]
        )
    handles, labels = ax.get_legend_handles_labels()
    n_algorithms = len(plot_df['algorithm'].unique())
    if handles:
        # ax.legend(handles[:n_algorithms], labels[:n_algorithms], title='Algorithm')
        new_labels = ["MORL", "SORL"]
        leg = ax.legend(
            handles[:n_algorithms],
            new_labels,
            title='Algorithm',
            frameon=True,
            title_fontproperties={'weight': 'bold', 'size': 20},
            prop={'weight': 'bold', 'size': 18},
            handlelength=1.9,
            handleheight=0.8,
            handletextpad=0.6,
            borderpad=0.7,
            labelspacing=0.4,
            loc='center',
            bbox_to_anchor=(0.95, 0.23),
        )
        for handle in leg.legend_handles:  # legendHandles   legend_handles
            handle.set_linewidth(1.5)
            handle.set_edgecolor("black")
        frame = leg.get_frame()
        frame.set_linewidth(2.0)
        frame.set_edgecolor("#333333")
    ax = plt.gca()
    # ax.set_xticklabels([
    #     metric_display_name.get(m.get_text(), m.get_text())
    #     for m in ax.get_xticklabels()
    # ])
    ticks = ax.get_xticks()
    old_labels = [t.get_text() for t in ax.get_xticklabels()]
    new_labels = [
        metric_display_name.get(l, l)
        for l in old_labels
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(new_labels)
    # for label in ax.get_xticklabels():
    #     text = label.get_text()
    #     if text in [
    #         r'$\#\mathit{UB}$',
    #         r'$\mathit{UBD}$',
    #         r'$\mathit{WBD}$',
    #         r'$\mathit{SCD}$'
    #     ]:
    #         x = label.get_position()[0]
    #         trans = ax.get_xaxis_transform()
    #         inv = plt.gcf().transFigure.inverted()
    #         x_fig, y_fig = inv.transform(trans.transform((x, 0)))
    #         plt.gcf().lines.append(
    #             plt.Line2D(
    #                 [x_fig - 0.015, x_fig + 0.015],
    #                 [y_fig - 0.08, y_fig - 0.08],
    #                 transform=plt.gcf().transFigure,
    #                 color='black',
    #                 linewidth=2
    #             )
    #         )
    draw_group_brace(
        ax,
        [
            r'$\mathit{OV}$',
            r'$\#\mathit{SV}$',
            r'$\mathit{SVS}$',
            r'$\#\mathit{MV}$',
            r'$\mathit{MVS}$',
            r'$\#\mathit{C}$',
        ],
        y_offset=0.1,
        label=r'$\mathit{effectiveness\ metric}$'
    )
    draw_group_brace(
        ax,
        [
            r'$\#\mathit{UB}$',
            r'$\mathit{UBD}$',
            r'$\mathit{WBD}$',
        ],
        y_offset=0.1,
        label=r'$\mathit{behavior\ diversity}$'
    )
    draw_group_brace(
        ax,
        [
            r'$\mathit{SCD}$',
        ],
        y_offset=0.1,
        label=r'$\mathit{scenario\ diversity}$'
    )
    draw_group_brace(
        ax,
        [
            r'$\#\mathit{UB}$',
            r'$\mathit{UBD}$',
            r'$\mathit{WBD}$',
            r'$\mathit{SCD}$',
        ],
        y_offset=0.2,
        label=r'$\mathit{diversity\ metric}$'
    )

    sns.despine(top=True, right=True, left=False, bottom=False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(3.0)
        ax.spines[spine].set_edgecolor('#333333')
    plt.ylim(0, 1)
    plt.xlabel("Metric", fontweight='bold', fontsize=33, labelpad=40)
    plt.ylabel("Normalized Value", fontweight='bold', fontsize=33)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    ax = plt.gca()
    labels = ax.get_xticklabels()
    for label in labels:
        label.set_path_effects([
            path_effects.withStroke(linewidth=0.7, foreground='black'),
            path_effects.Normal()
        ])
    plt.title("")
    plt.savefig(f'{root_dir}/RQ1.png', format="png", bbox_inches='tight')
    plt.savefig(f'{root_dir}/RQ1.pdf', format="pdf", bbox_inches='tight')
    plt.clf()
    # plt.tight_layout()
    # plt.show()

    # alg_pairs = list(itertools.combinations(algorithm_list, 2))
    # n_pairs = len(alg_pairs)
    # rows = []
    # for metric, alg_data in raw_stats_data.items():
    #     if metric.endswith('_num'):
    #         fisher_p_list = []
    #         fisher_or_list = []
    #         for a1, a2 in alg_pairs:
    #             v1 = alg_data.get(a1)
    #             v2 = alg_data.get(a2)
    #             if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
    #                 fisher_p_list.append(None)
    #                 fisher_or_list.append(None)
    #                 continue
    #             val1 = int(np.sum(v1))
    #             val2 = int(np.sum(v2))
    #             if not (0 <= val1 <= len(v1) * 100 and 0 <= val2 <= len(v2) * 100):
    #                 fisher_p_list.append(None)
    #                 fisher_or_list.append(None)
    #                 continue
    #             table = [[val1, len(v1) * 100 - val1], [val2, len(v2) * 100 - val2]]
    #             oratio, p_val = fisher_exact(table, alternative='two-sided')
    #             fisher_p_list.append(p_val)
    #             fisher_or_list.append(oratio)
    #         valid_idx = [i for i, p in enumerate(fisher_p_list) if p is not None]
    #         corrected_p = [None] * n_pairs
    #         if len(valid_idx) > 1:
    #             valid_p = [fisher_p_list[i] for i in valid_idx]
    #             _, corr_valid_p, _, _ = multipletests(
    #                 valid_p, alpha=0.05, method='holm'
    #             )
    #             for vi, pv in zip(valid_idx, corr_valid_p):
    #                 corrected_p[vi] = pv
    #         else:
    #             for vi in valid_idx:
    #                 corrected_p[vi] = fisher_p_list[vi]
    #         for (a1, a2), p, o in zip(alg_pairs, corrected_p, fisher_or_list):
    #             rows.append({
    #                 'metric': metric,
    #                 'algorithm_1': a1,
    #                 'algorithm_2': a2,
    #                 'p_value': p,
    #                 'p_value_format': significance_label(p),
    #                 'effect_size': None if o is None else round(float(o), 3),
    #                 'effect_size_format': None
    #             })
    #     else:
    #         mw_p_list = []
    #         mw_a12_list = []
    #         for a1, a2 in alg_pairs:
    #             g1 = alg_data.get(a1)
    #             g2 = alg_data.get(a2)
    #             if g1 is None or g2 is None or len(g1) < 10 or len(g2) < 10:
    #                 mw_p_list.append(None)
    #                 mw_a12_list.append(None)
    #                 continue
    #             u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
    #             a12 = u_stat / (len(g1) * len(g2))
    #             mw_p_list.append(p_val)
    #             mw_a12_list.append(a12)
    #         valid_idx = [i for i, p in enumerate(mw_p_list) if p is not None]
    #         corrected_p = [None] * n_pairs
    #         if len(valid_idx) > 1:
    #             valid_p = [mw_p_list[i] for i in valid_idx]
    #             _, corr_valid_p, _, _ = multipletests(
    #                 valid_p, alpha=0.05, method='holm'
    #             )
    #             for vi, pv in zip(valid_idx, corr_valid_p):
    #                 corrected_p[vi] = pv
    #         else:
    #             for vi in valid_idx:
    #                 corrected_p[vi] = mw_p_list[vi]
    #         for (a1, a2), p, a12 in zip(alg_pairs, corrected_p, mw_a12_list):
    #             rows.append({
    #                 'metric': metric,
    #                 'algorithm_1': a1,
    #                 'algorithm_2': a2,
    #                 'p_value': p,
    #                 'p_value_format': significance_label(p),
    #                 'effect_size': None if a12 is None else round(float(a12), 3),
    #                 'effect_size_format': a12_effect_label(a12)
    #             })
    # fieldnames = [
    #     'metric',
    #     'algorithm_1',
    #     'algorithm_2',
    #     'p_value',
    #     'p_value_format',
    #     'effect_size',
    #     'effect_size_format'
    # ]
    # with open(f'{root_dir}/RQ1.csv', 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.DictWriter(f, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for r in rows:
    #         writer.writerow(r)


def draw_group_brace_y(ax, target_texts, x_offset, label):
    fig = plt.gcf()
    ys = []
    for tick in ax.get_yticklabels():
        text = tick.get_text()
        if text in target_texts:
            y = tick.get_position()[1]
            trans = ax.get_yaxis_transform()
            inv = fig.transFigure.inverted()
            x_fig, y_fig = inv.transform(trans.transform((0, y)))
            ys.append(y_fig)
    if not ys:
        return
    x = x_fig - x_offset

    if len(ys) == 1:
        y = ys[0]
        fig.lines.append(
            plt.Line2D(
                [x + 0.024, x + 0.015],
                [y, y],
                transform=fig.transFigure,
                color='black',
                linewidth=1
            )
        )
        txt = fig.text(
            x - 0.005, y,
            label,
            ha='center',
            va='center',
            fontsize=13,
            rotation=90
        )
        return

    y_min, y_max = min(ys), max(ys)
    y_mid = (y_min + y_max) / 2
    renderer = fig.canvas.get_renderer()
    tmp_text = fig.text(
        0, 0, label,
        fontsize=13,
        rotation=90
    )
    bbox = tmp_text.get_window_extent(renderer=renderer)
    tmp_text.remove()
    fig_height_px = fig.get_size_inches()[1] * fig.dpi
    text_height = bbox.height / fig_height_px
    gap = text_height / 2 + 0.008
    fig.lines.append(
        plt.Line2D(
            [x, x],
            [y_min, y_mid - gap],
            transform=fig.transFigure,
            color='black',
            linewidth=1
        )
    )
    fig.lines.append(
        plt.Line2D(
            [x, x],
            [y_mid + gap, y_max],
            transform=fig.transFigure,
            color='black',
            linewidth=1
        )
    )
    fig.lines.append(
        plt.Line2D(
            [x, x + 0.022],
            [y_min, y_min],
            transform=fig.transFigure,
            color='black',
            linewidth=1
        )
    )
    fig.lines.append(
        plt.Line2D(
            [x, x + 0.022],
            [y_max, y_max],
            transform=fig.transFigure,
            color='black',
            linewidth=1
        )
    )
    txt = fig.text(
        x, y_mid,
        label,
        ha='center',
        va='center',
        fontsize=13,
        rotation=90,
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            pad=1.5
        )
    )


def RQ2_heatmap(root_dir, algorithm_list, objective_list, scenario_list):
    custom_row = ["tol_x1_nor", "tol_x2_nor", "x1_num", "x1_num_x1_nor", "x2_num", "x2_num_x2_nor", "x1_x2_num",
                  "x1_x2_x1_nor", "x1_x2_x2_nor", "col_num", "unique_action_set_num", "unique_action_diff_num_no",
                  "unique_action_diff", "all_scenario_diff"]
    metrics_process = {
        'x1_num': lambda x: x / 100,
        'x2_num': lambda x: x / 100,
        'x1_x2_num': lambda x: x / 100,
        'col_num': lambda x: x / 100,
        'unique_action_set_num': lambda x: x / 100,
        'unique_action_diff_num_no': lambda x: x / 5,
        'unique_action_diff': lambda x: x / 5,
        'all_scenario_diff': lambda x: x / (x + 1),
    }
    obj_pairs = []
    for i in range(len(objective_list) - 1):
        for j in range(i + 1, len(objective_list)):
            obj_pairs.append(f"{objective_list[i]}_{objective_list[j]}")
    heatmap_data = {metric: {op: {} for op in obj_pairs} for metric in custom_row}
    for i in range(len(objective_list) - 1):
        obj1 = objective_list[i]
        for j in range(i + 1, len(objective_list)):
            obj2 = objective_list[j]
            obj_pair = f"{obj1}_{obj2}"
            for algorithm in algorithm_list:
                temp_metric_data = {m: [] for m in custom_row}
                for scenario in scenario_list:
                    if algorithm == 'random':
                        folder_path = f"{root_dir}/{algorithm}"
                    else:
                        folder_path = f"{root_dir}/{algorithm}/eval/{obj1}_{obj2}"
                    f_path = next(
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if os.path.isdir(os.path.join(folder_path, f)) and scenario in f
                    )
                    pkl_path = os.path.join(f_path, 'metrics.pkl')
                    with open(pkl_path, 'rb') as f:
                        data_dict = pickle.load(f)
                    for metric in custom_row:
                        if metric in data_dict:
                            temp_metric_data[metric].extend(data_dict[metric])
                for metric in custom_row:
                    values = np.array(temp_metric_data[metric], dtype=float)
                    values = values[np.isfinite(values)]
                    if metric in metrics_process:
                        values = metrics_process[metric](values)
                    mean_value = values.mean() if len(values) > 0 else np.nan
                    heatmap_data[metric][obj_pair][algorithm] = mean_value

    alg_pairs = list(itertools.combinations(algorithm_list, 2))
    for algo_A, algo_B in alg_pairs:
        heatmap_matrix = np.zeros((len(custom_row), len(obj_pairs)))
        for i, metric in enumerate(custom_row):
            for j, obj_pair in enumerate(obj_pairs):
                heatmap_matrix[i, j] = (
                        heatmap_data[metric][obj_pair][algo_A]
                        - heatmap_data[metric][obj_pair][algo_B]
                )
        plt.figure(figsize=(12, 10))
        max_abs = np.nanmax(np.abs(heatmap_matrix))
        if max_abs == 0:
            max_abs = 1.0
        ax = sns.heatmap(
            heatmap_matrix,
            cmap="RdBu",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            annot=True,
            fmt=".3f",
            annot_kws={
                'size': 12,
                # 'weight': 'bold',
            },
            linewidths=0.7,
            linecolor='white',
            cbar_kws={'label': ''},
            xticklabels=['D – T', 'D – R', 'D – J', 'D – S', 'T – R', 'T – J', 'T – S', 'R – J', 'R – S', 'J – S'],
            yticklabels=[r'$\mathit{OV_{x1}}$', r'$\mathit{OV_{x2}}$', r'$\#\mathit{SV_{x1}}$', r'$\mathit{SVS_{x1}}$',
                         r'$\#\mathit{SV_{x2}}$', r'$\mathit{SVS_{x2}}$', r'$\#\mathit{MV_{x1,x2}}$',
                         r'$\mathit{MVS_{x1,x2}^{x1}}$', r'$\mathit{MVS_{x1,x2}^{x2}}$', r'$\#\mathit{C}$',
                         r'$\#\mathit{UB}$', r'$\mathit{UBD}$', r'$\mathit{WBD}$', r'$\mathit{SCD}$'],
            square=False
        )
        ax.set_facecolor('lightgray')
        cbar = ax.collections[0].colorbar
        # cbar.set_label(f'Performance Differences (MORL \u2013 SORL)',
        #                fontsize=15,
        #                # weight='bold',
        #                labelpad=17,
        #                rotation=270)
        cbar.ax.tick_params(labelsize=11)
        # for l in cbar.ax.get_yticklabels():
        #     l.set_fontweight('bold')
        for label in ax.get_yticklabels():
            # label.set_path_effects([
            #     path_effects.withStroke(linewidth=0.1, foreground='black'),
            #     path_effects.Normal()
            # ])
            label.set_fontsize(12)
        for label in ax.get_xticklabels():
            # label.set_path_effects([
            #     path_effects.withStroke(linewidth=0.1, foreground='black'),
            #     path_effects.Normal()
            # ])
            label.set_fontsize(12)

        draw_group_brace_y(
            ax,
            [
                r'$\mathit{OV_{x1}}$', r'$\mathit{OV_{x2}}$',
                r'$\#\mathit{SV_{x1}}$', r'$\mathit{SVS_{x1}}$',
                r'$\#\mathit{SV_{x2}}$', r'$\mathit{SVS_{x2}}$',
                r'$\#\mathit{MV_{x1,x2}}$',
                r'$\mathit{MVS_{x1,x2}^{x1}}$',
                r'$\mathit{MVS_{x1,x2}^{x2}}$',
                r'$\#\mathit{C}$'
            ],
            x_offset=0.09,
            label=r'$\mathit{effectiveness\ metric}$'
        )
        draw_group_brace_y(
            ax,
            [
                r'$\#\mathit{UB}$',
                r'$\mathit{UBD}$',
                r'$\mathit{WBD}$',
                r'$\mathit{SCD}$'
            ],
            x_offset=0.12,
            label=r'$\mathit{diversity}$'+'\n'+r'$\mathit{metric}$'
        )
        draw_group_brace_y(
            ax,
            [
                r'$\#\mathit{UB}$',
                r'$\mathit{UBD}$',
                r'$\mathit{WBD}$',
            ],
            x_offset=0.07,
            label=r'$\mathit{behavior}$'+'\n'+r'$\mathit{diversity}$'
        )
        draw_group_brace_y(
            ax,
            [
                r'$\mathit{SCD}$'
            ],
            x_offset=0.07,
            label=r'$\mathit{scenario}$'+'\n'+r'$\mathit{diversity}$'
        )

        plt.title("")
        plt.savefig(f'{root_dir}/RQ2_{algo_A}_{algo_B}.png', format="png", bbox_inches='tight')
        plt.savefig(f'{root_dir}/RQ2_{algo_A}_{algo_B}.pdf', format="pdf", bbox_inches='tight')
        # plt.tight_layout()
        # plt.show()


def RQ2_statistic(root_dir, algorithm_list, objective_list, scenario_list):
    custom_row = ["tol_x1", "tol_x2", "x1_num", "x1_num_x1", "x2_num", "x2_num_x2", "x1_x2_num", "x1_x2_x1", "x1_x2_x2",
                  "col_num", "unique_action_set_num", "unique_action_diff_num_no", "unique_action_diff", "all_scenario_diff"]
    obj_pairs = []
    for i in range(len(objective_list) - 1):
        for j in range(i + 1, len(objective_list)):
            obj_pairs.append(f"{objective_list[i]}_{objective_list[j]}")
    raw_data = {metric: {op: {} for op in obj_pairs} for metric in custom_row}
    for i in range(len(objective_list) - 1):
        obj1 = objective_list[i]
        for j in range(i + 1, len(objective_list)):
            obj2 = objective_list[j]
            obj_pair = f"{obj1}_{obj2}"
            for algorithm in algorithm_list:
                temp_metric_data = {m: [] for m in custom_row}
                for scenario in scenario_list:
                    if algorithm == 'random':
                        folder_path = f"{root_dir}/{algorithm}"
                    else:
                        folder_path = f"{root_dir}/{algorithm}/eval/{obj1}_{obj2}"
                    f_path = next(
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if os.path.isdir(os.path.join(folder_path, f)) and scenario in f
                    )
                    pkl_path = os.path.join(f_path, 'metrics.pkl')
                    with open(pkl_path, 'rb') as f:
                        data_dict = pickle.load(f)
                    for metric in custom_row:
                        if metric in data_dict:
                            temp_metric_data[metric].extend(data_dict[metric])
                for metric in custom_row:
                    values = np.array(temp_metric_data[metric], dtype=float)
                    values = values[np.isfinite(values)]
                    raw_data[metric][obj_pair][algorithm] = values.tolist()

    alg_pairs = list(itertools.combinations(algorithm_list, 2))
    n_pairs = len(alg_pairs)
    rows = []
    for metric, metric_data in raw_data.items():
        for obj_pair, alg_data in metric_data.items():
            if metric.endswith('_num'):
                fisher_p_list = []
                fisher_or_list = []
                for a1, a2 in alg_pairs:
                    v1 = alg_data.get(a1)
                    v2 = alg_data.get(a2)
                    if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
                        fisher_p_list.append(None)
                        fisher_or_list.append(None)
                        continue
                    val1 = int(np.sum(v1))
                    val2 = int(np.sum(v2))
                    if not (
                            0 <= val1 <= len(v1) * 100 and
                            0 <= val2 <= len(v2) * 100
                    ):
                        fisher_p_list.append(None)
                        fisher_or_list.append(None)
                        continue
                    table = [
                        [val1, len(v1) * 100 - val1],
                        [val2, len(v2) * 100 - val2]
                    ]
                    oratio, p_val = fisher_exact(table, alternative='two-sided')
                    fisher_p_list.append(p_val)
                    fisher_or_list.append(oratio)
                valid_idx = [i for i, p in enumerate(fisher_p_list) if p is not None]
                corrected_p = [None] * n_pairs
                if len(valid_idx) > 1:
                    valid_p = [fisher_p_list[i] for i in valid_idx]
                    _, corr_valid_p, _, _ = multipletests(
                        valid_p, alpha=0.05, method='holm'
                    )
                    for vi, pv in zip(valid_idx, corr_valid_p):
                        corrected_p[vi] = pv
                else:
                    for vi in valid_idx:
                        corrected_p[vi] = fisher_p_list[vi]
                for (a1, a2), p, o in zip(alg_pairs, corrected_p, fisher_or_list):
                    rows.append({
                        'metric': metric,
                        'obj_pair': obj_pair,
                        'algorithm_1': a1,
                        'algorithm_2': a2,
                        'p_value': p,
                        'p_value_format': significance_label(p),
                        'effect_size': None if o is None else round(float(o), 3),
                        'effect_size_format': None
                    })
            else:
                mw_p_list = []
                mw_a12_list = []
                for a1, a2 in alg_pairs:
                    g1 = alg_data.get(a1)
                    g2 = alg_data.get(a2)
                    if g1 is None or g2 is None or len(g1) < 10 or len(g2) < 10:
                        mw_p_list.append(None)
                        mw_a12_list.append(None)
                        continue
                    u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
                    a12 = u_stat / (len(g1) * len(g2))
                    mw_p_list.append(p_val)
                    mw_a12_list.append(a12)
                valid_idx = [i for i, p in enumerate(mw_p_list) if p is not None]
                corrected_p = [None] * n_pairs
                if len(valid_idx) > 1:
                    valid_p = [mw_p_list[i] for i in valid_idx]
                    _, corr_valid_p, _, _ = multipletests(
                        valid_p, alpha=0.05, method='holm'
                    )
                    for vi, pv in zip(valid_idx, corr_valid_p):
                        corrected_p[vi] = pv
                else:
                    for vi in valid_idx:
                        corrected_p[vi] = mw_p_list[vi]
                for (a1, a2), p, a12 in zip(alg_pairs, corrected_p, mw_a12_list):
                    rows.append({
                        'metric': metric,
                        'obj_pair': obj_pair,
                        'algorithm_1': a1,
                        'algorithm_2': a2,
                        'p_value': p,
                        'p_value_format': significance_label(p),
                        'effect_size': None if a12 is None else round(float(a12), 3),
                        'effect_size_format': a12_effect_label(a12)
                    })
    fieldnames = [
        'metric',
        'obj_pair',
        'algorithm_1',
        'algorithm_2',
        'p_value',
        'p_value_format',
        'effect_size',
        'effect_size_format'
    ]
    with open(f'{root_dir}/RQ2.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    wide_table = defaultdict(dict)
    for r in rows:
        metric = r['metric']
        obj_pair = r['obj_pair']
        p_col = f"{obj_pair}_p"
        e_col = f"{obj_pair}_e"
        wide_table[metric][p_col] = (
            'NA' if r['p_value_format'] is None else r['p_value_format']
        )
        wide_table[metric][e_col] = (
            'NA' if r['effect_size'] is None else r['effect_size']
        )
    obj_pair_order = [
        "distance_time_to_collision",
        "distance_completion",
        "distance_comfort",
        "distance_speed_diff",
        "time_to_collision_completion",
        "time_to_collision_comfort",
        "time_to_collision_speed_diff",
        "completion_comfort",
        "completion_speed_diff",
        "comfort_speed_diff"
    ]
    fieldnames = ['metric']
    for op in obj_pair_order:
        fieldnames.append(f"{op}_p")
        fieldnames.append(f"{op}_e")
    output_path = f"{root_dir}/RQ2_wide.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in custom_row:
            row = {'metric': metric}
            metric_data = wide_table.get(metric, {})
            for col in fieldnames[1:]:
                row[col] = metric_data.get(col, 'NA')
            writer.writerow(row)


def obtain_objective_value(data, file_name):
    data['episode'] = data['episode'].astype(int)
    episode_start = 0
    episode_end = data['episode'].iloc[-1]
    episode = episode_start

    dis_episode, ttc_episode, comp_episode, comf_episode, spdif_episode = [], [], [], [], []
    tol_dis, tol_ttc, tol_comp, tol_comf, tol_spdif = [], [], [], [], []

    has_dis = 'distance' in data.columns or 'distance' in file_name
    has_ttc = 'time_to_collision' in data.columns or 'time_to_collision' in file_name
    has_comp = 'completion' in data.columns or 'completion' in file_name
    has_comf = 'comfort' in data.columns or 'comfort' in file_name
    has_spdif = 'speed_diff' in data.columns or 'speed_diff' in file_name

    data.index = data.index.astype(int)
    for i in range(data[data['episode'] == episode_start].index.min(),
                   data[data['episode'] == episode_end].index.max() + 2):
        if i == len(data) or data['episode'].iloc[i] != episode:
            if data['episode'].iloc[i - 1] >= episode_end - 99:
                if has_dis: tol_dis.append(np.mean(dis_episode))
                if has_ttc: tol_ttc.append(np.mean(ttc_episode))
                if has_comp: tol_comp.append(sum(comp_episode))
                if has_comf: tol_comf.append(np.mean(comf_episode))
                if has_spdif: tol_spdif.append(np.mean(spdif_episode))
            if i == len(data):
                break
            episode = data['episode'].iloc[i]
            dis_episode, ttc_episode, comp_episode, comf_episode, spdif_episode = [], [], [], [], []

        if has_dis: dis_episode.append(data['distance'].iloc[i])
        if has_ttc: ttc_episode.append(data['time_to_collision'].iloc[i])
        if has_comp: comp_episode.append(data['completion'].iloc[i])
        if has_comf: comf_episode.append(data['comfort'].iloc[i])
        if has_spdif: spdif_episode.append(data['speed_diff'].iloc[i])

    metrics_dict = {}
    if has_dis: metrics_dict['distance'] = tol_dis
    if has_ttc: metrics_dict['time_to_collision'] = tol_ttc
    if has_comp: metrics_dict['completion'] = tol_comp
    if has_comf: metrics_dict['comfort'] = tol_comf
    if has_spdif: metrics_dict['speed_diff'] = tol_spdif

    return metrics_dict


def RQ2_objective_correlation_spearman(root_dir, algorithm_list, objective_list, scenario_list):

    output_path = f"{root_dir}/RQ2_correlation_spearman.csv"
    output_format_path = f"{root_dir}/RQ2_correlation_spearman_format.csv"
    results = []
    results_format = []
    columns = ['scenario']
    overall_obj_pair_data = {}
    for i in range(len(objective_list) - 1):
        obj1 = objective_list[i]
        for j in range(i + 1, len(objective_list)):
            obj2 = objective_list[j]
            columns.extend([f"{obj1}_{obj2}_p", f"{obj1}_{obj2}_r"])
            overall_obj_pair_data[f"{obj1}_{obj2}"] = [[], []]

    for scenario in scenario_list:
        p_list = []
        rho_list = []
        row = [scenario]
        row_f = [scenario]

        for i in range(len(objective_list) - 1):
            obj1 = objective_list[i]
            for j in range(i + 1, len(objective_list)):
                obj2 = objective_list[j]
                obj1_values = []
                obj2_values = []

                for algorithm in algorithm_list:
                    if algorithm == 'random':
                        folder_path = f"{root_dir}/{algorithm}"
                    else:
                        folder_path = f"{root_dir}/{algorithm}/eval/{obj1}_{obj2}"

                    file_name = next(
                        file for file in os.listdir(folder_path)
                        if file.endswith('.csv') and scenario in file
                    )
                    f_name = os.path.splitext(file_name)[0]
                    full_path = os.path.join(folder_path, file_name)

                    data = pd.read_csv(full_path)
                    metrics = obtain_objective_value(data, f_name)

                    obj1_data = metrics.get(obj1, [])
                    obj2_data = metrics.get(obj2, [])
                    if obj1_data and obj2_data:
                        obj1_values.extend(obj1_data)
                        obj2_values.extend(obj2_data)
                        overall_obj_pair_data[f"{obj1}_{obj2}"][0].extend(obj1_data)
                        overall_obj_pair_data[f"{obj1}_{obj2}"][1].extend(obj2_data)

                rho, p_value = spearmanr(obj1_values, obj2_values)
                p_list.append(p_value)
                rho_list.append(rho)

        valid_idx = [
            i for i, p in enumerate(p_list)
            if p is not None and not (isinstance(p, float) and math.isnan(p))
        ]
        corrected_p_values = [None] * len(p_list)
        if len(valid_idx) > 1:
            valid_p = [p_list[i] for i in valid_idx]
            _, corr_valid_p, _, _ = multipletests(valid_p, alpha=0.05, method='holm')
            for i, pv in zip(valid_idx, corr_valid_p):
                corrected_p_values[i] = pv
        else:
            for i in valid_idx:
                corrected_p_values[i] = p_list[i]

        for p, r in zip(corrected_p_values, rho_list):
            if p is None or (isinstance(p, float) and math.isnan(p)):
                p_out = None
                r_out = None
                p_sign = 'NA'
                r_fmt = 'NA'
            else:
                p_out = p
                r_out = r
                if p < 0.01:
                    p_sign = '<0.01'
                elif p < 0.05:
                    p_sign = '<0.05'
                else:
                    p_sign = '>=0.05'
                r_fmt = round(r, 3) if r is not None else 'NA'
            row.extend([p_out, r_out])
            row_f.extend([p_sign, r_fmt])

        results.append(row)
        results_format.append(row_f)

    row_overall = ['overall']
    row_f_overall = ['overall']
    p_list_overall = []
    rho_list_overall = []

    for obj_pair_key, (obj1_vals, obj2_vals) in overall_obj_pair_data.items():
        rho, p_value = spearmanr(obj1_vals, obj2_vals)
        p_list_overall.append(p_value)
        rho_list_overall.append(rho)

    valid_idx = [
        i for i, p in enumerate(p_list_overall)
        if p is not None and not (isinstance(p, float) and math.isnan(p))
    ]
    corrected_p_values_overall = [None] * len(p_list_overall)
    if len(valid_idx) > 1:
        valid_p = [p_list_overall[i] for i in valid_idx]
        _, corr_valid_p, _, _ = multipletests(valid_p, alpha=0.05, method='holm')
        for i, pv in zip(valid_idx, corr_valid_p):
            corrected_p_values_overall[i] = pv
    else:
        for i in valid_idx:
            corrected_p_values_overall[i] = p_list_overall[i]

    for p, r in zip(corrected_p_values_overall, rho_list_overall):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            p_out = None
            r_out = None
            p_sign = 'NA'
            r_fmt = 'NA'
        else:
            p_out = p
            r_out = r
            if p < 0.01:
                p_sign = '<0.01'
            elif p < 0.05:
                p_sign = '<0.05'
            else:
                p_sign = '>=0.05'
            r_fmt = round(r, 3) if r is not None else 'NA'

        row_overall.extend([p_out, r_out])
        row_f_overall.extend([p_sign, r_fmt])

    results.append(row_overall)
    results_format.append(row_f_overall)

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
    df_f = pd.DataFrame(results_format, columns=columns)
    df_f.to_csv(output_format_path, index=False)


def RQ3_heatmap_merge(root_dir, algorithm_list, objective_list, scenario_list):
    custom_row = ["tol_x1_nor", "tol_x2_nor", "x1_num", "x1_num_x1_nor", "x2_num", "x2_num_x2_nor", "x1_x2_num",
                  "x1_x2_x1_nor", "x1_x2_x2_nor", "col_num", "unique_action_set_num", "unique_action_diff_num_no",
                  "unique_action_diff", "all_scenario_diff"]
    metrics_process = {
        'x1_num': lambda x: x / 100,
        'x2_num': lambda x: x / 100,
        'x1_x2_num': lambda x: x / 100,
        'col_num': lambda x: x / 100,
        'unique_action_set_num': lambda x: x / 100,
        'unique_action_diff_num_no': lambda x: x / 5,
        'unique_action_diff': lambda x: x / 5,
        'all_scenario_diff': lambda x: x / (x + 1),
    }
    merge_map = {
        "tol_merge": ["tol_x1_nor", "tol_x2_nor"],
        "x_num_merge": ["x1_num", "x2_num"],
        "x_num_x_merge": ["x1_num_x1_nor", "x2_num_x2_nor"],
        "x12_nor_merge": ["x1_x2_x1_nor", "x1_x2_x2_nor"],
    }
    merged_row = ["tol_merge", "x_num_merge", "x_num_x_merge", "x1_x2_num", "x12_nor_merge", "col_num",
                  "unique_action_set_num", "unique_action_diff_num_no", "unique_action_diff", "all_scenario_diff"]
    heatmap_data = {metric: {sc: {} for sc in scenario_list} for metric in merged_row}
    for scenario in scenario_list:
        for algorithm in algorithm_list:
            temp_metric_data = {m: [] for m in custom_row}
            for i in range(len(objective_list) - 1):
                obj1 = objective_list[i]
                for j in range(i + 1, len(objective_list)):
                    obj2 = objective_list[j]
                    if algorithm == 'random':
                        folder_path = f"{root_dir}/{algorithm}"
                    else:
                        folder_path = f"{root_dir}/{algorithm}/eval/{obj1}_{obj2}"
                    f_path = next(
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if os.path.isdir(os.path.join(folder_path, f)) and scenario in f
                    )
                    pkl_path = os.path.join(f_path, 'metrics.pkl')
                    with open(pkl_path, 'rb') as f:
                        data_dict = pickle.load(f)
                    for metric in custom_row:
                        if metric in data_dict:
                            temp_metric_data[metric].extend(data_dict[metric])
            processed_metric = {}
            for metric in custom_row:
                values = np.array(temp_metric_data[metric], dtype=float)
                values = values[np.isfinite(values)]
                if metric in metrics_process:
                    values = metrics_process[metric](values)
                processed_metric[metric] = values
            for merge_name, metric_list in merge_map.items():
                merged_values = []
                for m in metric_list:
                    merged_values.extend(processed_metric.get(m, []))
                merged_values = np.array(merged_values)
                mean_value = merged_values.mean() if len(merged_values) > 0 else np.nan
                heatmap_data[merge_name][scenario][algorithm] = mean_value
            for metric in merged_row:
                if metric in merge_map:
                    continue
                values = processed_metric.get(metric, [])
                mean_value = values.mean() if len(values) > 0 else np.nan
                heatmap_data[metric][scenario][algorithm] = mean_value

    alg_pairs = list(itertools.combinations(algorithm_list, 2))
    for algo_A, algo_B in alg_pairs:
        heatmap_matrix = np.zeros((len(merged_row), len(scenario_list)))
        for i, metric in enumerate(merged_row):
            for j, scenario in enumerate(scenario_list):
                heatmap_matrix[i, j] = (
                        heatmap_data[metric][scenario][algo_A]
                        - heatmap_data[metric][scenario][algo_B]
                )
        plt.figure(figsize=(12, 8))
        max_abs = np.nanmax(np.abs(heatmap_matrix))
        if max_abs == 0:
            max_abs = 1.0
        ax = sns.heatmap(
            heatmap_matrix,
            cmap="RdBu",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            annot=True,
            fmt=".3f",
            annot_kws={
                'size': 12,
                # 'weight': 'bold',
            },
            linewidths=0.7,
            linecolor='white',
            cbar_kws={'label': ''},
            xticklabels=[r'$\mathit{Road1}$', r'$\mathit{Road2}$', r'$\mathit{Road3}$', r'$\mathit{Road4}$',
                         r'$\mathit{Road5}$', r'$\mathit{Road6}$'],
            yticklabels=[r'$\mathit{OV}$', r'$\#\mathit{SV}$', r'$\mathit{SVS}$', r'$\#\mathit{MV}$',
                         r'$\mathit{MVS}$', r'$\#\mathit{C}$', r'$\#\mathit{UB}$', r'$\mathit{UBD}$',
                         r'$\mathit{WBD}$', r'$\mathit{SCD}$'],
            square=False
        )
        ax.set_facecolor('lightgray')
        cbar = ax.collections[0].colorbar
        # cbar.set_label(f'Performance Differences (MORL \u2013 SORL)',
        #                fontsize=15,
        #                # weight='bold',
        #                labelpad=17,
        #                rotation=270)
        cbar.ax.tick_params(labelsize=11)
        # for l in cbar.ax.get_yticklabels():
        #     l.set_fontweight('bold')
        for label in ax.get_yticklabels():
            # label.set_path_effects([
            #     path_effects.withStroke(linewidth=0.1, foreground='black'),
            #     path_effects.Normal()
            # ])
            label.set_fontsize(12)
            label.set_rotation(0)
        for label in ax.get_xticklabels():
            # label.set_path_effects([
            #     path_effects.withStroke(linewidth=0.1, foreground='black'),
            #     path_effects.Normal()
            # ])
            label.set_fontsize(12)

        draw_group_brace_y(
            ax,
            [
                r'$\mathit{OV}$',
                r'$\#\mathit{SV}$', r'$\mathit{SVS}$',
                r'$\#\mathit{MV}$', r'$\mathit{MVS}$',
                r'$\#\mathit{C}$'
            ],
            x_offset=0.07,
            label=r'$\mathit{effectiveness\ metric}$'
        )
        draw_group_brace_y(
            ax,
            [
                r'$\#\mathit{UB}$',
                r'$\mathit{UBD}$',
                r'$\mathit{WBD}$',
                r'$\mathit{SCD}$'
            ],
            x_offset=0.12,
            label=r'$\mathit{diversity}$'+'\n'+r'$\mathit{metric}$'
        )
        draw_group_brace_y(
            ax,
            [
                r'$\#\mathit{UB}$',
                r'$\mathit{UBD}$',
                r'$\mathit{WBD}$',
            ],
            x_offset=0.07,
            label=r'$\mathit{behavior}$'+'\n'+r'$\mathit{diversity}$'
        )
        draw_group_brace_y(
            ax,
            [
                r'$\mathit{SCD}$'
            ],
            x_offset=0.07,
            label=r'$\mathit{scenario}$'+'\n'+r'$\mathit{diversity}$'
        )

        plt.title("")
        plt.savefig(f'{root_dir}/RQ3_{algo_A}_{algo_B}.png', format="png", bbox_inches='tight')
        plt.savefig(f'{root_dir}/RQ3_{algo_A}_{algo_B}.pdf', format="pdf", bbox_inches='tight')
        # plt.tight_layout()
        # plt.show()


def RQ3_statistic_merge(root_dir, algorithm_list, objective_list, scenario_list):
    custom_row = ["tol_x1_nor", "tol_x2_nor", "x1_num", "x1_num_x1_nor", "x2_num", "x2_num_x2_nor", "x1_x2_num",
                  "x1_x2_x1_nor", "x1_x2_x2_nor", "col_num", "unique_action_set_num", "unique_action_diff_num_no",
                  "unique_action_diff", "all_scenario_diff"]
    metrics_process = {
        'x1_num': lambda x: x / 100,
        'x2_num': lambda x: x / 100,
        'x1_x2_num': lambda x: x / 100,
        'col_num': lambda x: x / 100,
        'unique_action_set_num': lambda x: x / 100,
        'unique_action_diff_num_no': lambda x: x / 5,
        'unique_action_diff': lambda x: x / 5,
        'all_scenario_diff': lambda x: x / (x + 1),
    }
    merge_map = {
        "tol_merge": ["tol_x1_nor", "tol_x2_nor"],
        "x_num_merge": ["x1_num", "x2_num"],
        "x_num_x_merge": ["x1_num_x1_nor", "x2_num_x2_nor"],
        "x12_nor_merge": ["x1_x2_x1_nor", "x1_x2_x2_nor"],
    }
    merged_row = ["tol_merge", "x_num_merge", "x_num_x_merge", "x1_x2_num", "x12_nor_merge", "col_num",
                  "unique_action_set_num", "unique_action_diff_num_no", "unique_action_diff", "all_scenario_diff"]
    merged_data = {metric: {sc: {} for sc in scenario_list} for metric in merged_row}
    for scenario in scenario_list:
        for algorithm in algorithm_list:
            temp_metric_data = {m: [] for m in custom_row}
            for i in range(len(objective_list) - 1):
                obj1 = objective_list[i]
                for j in range(i + 1, len(objective_list)):
                    obj2 = objective_list[j]
                    if algorithm == 'random':
                        folder_path = f"{root_dir}/{algorithm}"
                    else:
                        folder_path = f"{root_dir}/{algorithm}/eval/{obj1}_{obj2}"
                    f_path = next(
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if os.path.isdir(os.path.join(folder_path, f)) and scenario in f
                    )
                    pkl_path = os.path.join(f_path, 'metrics.pkl')
                    with open(pkl_path, 'rb') as f:
                        data_dict = pickle.load(f)
                    for metric in custom_row:
                        if metric in data_dict:
                            temp_metric_data[metric].extend(data_dict[metric])
            processed_metric = {}
            for metric in custom_row:
                values = np.array(temp_metric_data[metric], dtype=float)
                values = values[np.isfinite(values)]
                if metric in metrics_process:
                    values = metrics_process[metric](values)
                processed_metric[metric] = values.tolist()
            for merge_name, metric_list in merge_map.items():
                merged_values = []
                for m in metric_list:
                    merged_values.extend(processed_metric.get(m, []))
                merged_data[merge_name][scenario][algorithm] = merged_values
            for metric in merged_row:
                if metric in merge_map:
                    continue
                merged_data[metric][scenario][algorithm] = processed_metric.get(metric, [])

    alg_pairs = list(itertools.combinations(algorithm_list, 2))
    n_pairs = len(alg_pairs)
    rows = []
    for metric, metric_data in merged_data.items():
        for scenario, alg_data in metric_data.items():
            if metric.endswith('_num'):
                fisher_p_list = []
                fisher_or_list = []
                for a1, a2 in alg_pairs:
                    v1 = alg_data.get(a1)
                    v2 = alg_data.get(a2)
                    if v1 is None or v2 is None or len(v1) == 0 or len(v2) == 0:
                        fisher_p_list.append(None)
                        fisher_or_list.append(None)
                        continue
                    val1 = int(np.sum(v1))
                    val2 = int(np.sum(v2))
                    if not (
                            0 <= val1 <= len(v1) * 100 and
                            0 <= val2 <= len(v2) * 100
                    ):
                        fisher_p_list.append(None)
                        fisher_or_list.append(None)
                        continue
                    table = [
                        [val1, len(v1) * 100 - val1],
                        [val2, len(v2) * 100 - val2]
                    ]
                    oratio, p_val = fisher_exact(table, alternative='two-sided')
                    fisher_p_list.append(p_val)
                    fisher_or_list.append(oratio)
                valid_idx = [i for i, p in enumerate(fisher_p_list) if p is not None]
                corrected_p = [None] * n_pairs
                if len(valid_idx) > 1:
                    valid_p = [fisher_p_list[i] for i in valid_idx]
                    _, corr_valid_p, _, _ = multipletests(
                        valid_p, alpha=0.05, method='holm'
                    )
                    for vi, pv in zip(valid_idx, corr_valid_p):
                        corrected_p[vi] = pv
                else:
                    for vi in valid_idx:
                        corrected_p[vi] = fisher_p_list[vi]
                for (a1, a2), p, o in zip(alg_pairs, corrected_p, fisher_or_list):
                    rows.append({
                        'metric': metric,
                        'scenario': scenario,
                        'algorithm_1': a1,
                        'algorithm_2': a2,
                        'p_value': p,
                        'p_value_format': significance_label(p),
                        'effect_size': None if o is None else round(float(o), 3),
                        'effect_size_format': None
                    })
            else:
                mw_p_list = []
                mw_a12_list = []
                for a1, a2 in alg_pairs:
                    g1 = alg_data.get(a1)
                    g2 = alg_data.get(a2)
                    if g1 is None or g2 is None or len(g1) < 10 or len(g2) < 10:
                        mw_p_list.append(None)
                        mw_a12_list.append(None)
                        continue
                    u_stat, p_val = mannwhitneyu(g1, g2, alternative='two-sided')
                    a12 = u_stat / (len(g1) * len(g2))
                    mw_p_list.append(p_val)
                    mw_a12_list.append(a12)
                valid_idx = [i for i, p in enumerate(mw_p_list) if p is not None]
                corrected_p = [None] * n_pairs
                if len(valid_idx) > 1:
                    valid_p = [mw_p_list[i] for i in valid_idx]
                    _, corr_valid_p, _, _ = multipletests(
                        valid_p, alpha=0.05, method='holm'
                    )
                    for vi, pv in zip(valid_idx, corr_valid_p):
                        corrected_p[vi] = pv
                else:
                    for vi in valid_idx:
                        corrected_p[vi] = mw_p_list[vi]
                for (a1, a2), p, a12 in zip(alg_pairs, corrected_p, mw_a12_list):
                    rows.append({
                        'metric': metric,
                        'scenario': scenario,
                        'algorithm_1': a1,
                        'algorithm_2': a2,
                        'p_value': p,
                        'p_value_format': significance_label(p),
                        'effect_size': None if a12 is None else round(float(a12), 3),
                        'effect_size_format': a12_effect_label(a12)
                    })
    fieldnames = [
        'metric',
        'scenario',
        'algorithm_1',
        'algorithm_2',
        'p_value',
        'p_value_format',
        'effect_size',
        'effect_size_format'
    ]
    with open(f'{root_dir}/RQ3.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    wide_table = defaultdict(dict)
    for r in rows:
        metric = r['metric']
        scenario = r['scenario']
        p_col = f"{scenario}_p"
        e_col = f"{scenario}_e"
        wide_table[metric][p_col] = (
            'NA' if r['p_value_format'] is None else r['p_value_format']
        )
        wide_table[metric][e_col] = (
            'NA' if r['effect_size'] is None else r['effect_size']
        )
    fieldnames = ['metric']
    for sc in scenario_list:
        fieldnames.append(f"{sc}_p")
        fieldnames.append(f"{sc}_e")
    output_path = f"{root_dir}/RQ3_wide.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in merged_row:
            row = {'metric': metric}
            metric_data = wide_table.get(metric, {})
            for col in fieldnames[1:]:
                row[col] = metric_data.get(col, 'NA')
            writer.writerow(row)


if __name__ == '__main__':

    root_dir = './eval_results/interfuser'
    # algorithm_list = ['envelope', 'single', 'random']
    algorithm_list = ['envelope', 'single']
    objective_list = ['distance', 'time_to_collision', 'completion', 'comfort', 'speed_diff']
    # objective_list = ['comfort', 'speed_diff']
    scenario_list = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'scenario_6']

    generate_metric_csv(root_dir, algorithm_list, objective_list, scenario_list)

    RQ1_box_plot_statistic(root_dir, algorithm_list, objective_list, scenario_list)

    RQ2_heatmap(root_dir, algorithm_list, objective_list, scenario_list)

    RQ2_statistic(root_dir, algorithm_list, objective_list, scenario_list)

    RQ2_objective_correlation_spearman(root_dir, algorithm_list, objective_list, scenario_list)

    RQ3_heatmap_merge(root_dir, algorithm_list, objective_list, scenario_list)

    RQ3_statistic_merge(root_dir, algorithm_list, objective_list, scenario_list)
