import os
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager
import sys

sys.path.append('..')

from core.machine import MachineConfig
from playground.Non_DAG.algorithm.random_algorithm import RandomAlgorithm
from playground.Non_DAG.algorithm.tetris import Tetris
from playground.Non_DAG.algorithm.first_fit import FirstFitAlgorithm
from playground.Non_DAG.algorithm.DeepJS.DRL import RLAlgorithm
from playground.Non_DAG.algorithm.DeepJS.agent import Agent
from playground.Non_DAG.algorithm.DeepJS.brain import Brain

from playground.Non_DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver, AverageCompletionRewardGiver, AverageSlowDownRewardGiver

from playground.Non_DAG.utils.csv_reader import CSVReader
from playground.Non_DAG.utils.feature_functions import features_extract_func, features_normalize_func
from playground.Non_DAG.utils.tools import multiprocessing_run, average_completion, average_slowdown
from playground.Non_DAG.utils.episode import Episode
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = ''

np.random.seed(41)
tf.random.set_random_seed(41)
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 10
n_iter = 200
# n_iter = 2
n_episode = 12
# jobs_csv = '../jobs_files/jobs.csv'
jobs_csv = '../jobs_files/jobs_2017.csv'

brain = Brain(6)
# reward_giver = MakespanRewardGiver(-1)
reward_giver = AverageCompletionRewardGiver()
features_extract_func = features_extract_func
features_normalize_func = features_normalize_func

name = '%s-%s-m%d' % (reward_giver.name, brain.name, machines_number)
model_dir = './agents/%s' % name

train_info_dir = './agents/training/avgCompletionReward'
# train_info_dir = "/content/drive/MyDrive/GoogleDrive/MyRepo/"
# ************************ Parameters Setting End ************************

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

# agent = Agent(name, brain, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
#               model_save_path='%s/model.ckpt' % model_dir)
agent = Agent(name, brain, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
              model_save_path='%s/model.ckpt' % train_info_dir)

machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
jobs_configs = csv_reader.generate(0, jobs_len)


def set_path():
    from dir_info import root_dir_abs
    path = root_dir_abs()
    os.environ['PYTHONPATH'] = root_dir_abs()


# def save_train_info(agent: Agent, itr: int):
#     if not os.path.exists(train_info_dir):
#         os.makedirs(train_info_dir, exist_ok=True)
#     filename = 'chkpt_' + str(itr) + '.pkl'
#     filepath = os.path.join(train_info_dir, filename)
#     agent.save_chkpt(filepath)
def save_train_info(agent: Agent, itr: int, reward_type="mkspan"):
    if not os.path.exists(train_info_dir):
        os.makedirs(train_info_dir, exist_ok=True)
    filename = 'chkpt_' + str(itr) + "_" + reward_type + '.pkl'
    filepath = os.path.join(train_info_dir, filename)
    agent.save_chkpt(filepath)
    hist_name = f"hist_{reward_type}.csv"
    hist_path = os.path.join(train_info_dir, hist_name)
    df = pd.DataFrame(hist)
    df.to_csv(hist_path)
    print(f"save chkpt: {filename} | save hist: {hist_name}")


def algo_random():
    tic = time.time()
    algorithm = RandomAlgorithm()
    episode = Episode(machine_configs, jobs_configs, algorithm, None)
    episode.run()
    print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))


def algo_first_fit():
    tic = time.time()
    algorithm = FirstFitAlgorithm()
    episode = Episode(machine_configs, jobs_configs, algorithm, None)
    episode.run()
    print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))


def algo_tetris():
    tic = time.time()
    algorithm = Tetris()
    episode = Episode(machine_configs, jobs_configs, algorithm, None)
    episode.run()
    print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))


save_chkpt_every = 30


def train_algo_deep_js():
    for itr in range(n_iter):
        tic = time.time()
        print("********** DeepJS Iteration %i ************" % itr)
        processes = []

        manager = Manager()
        trajectories = manager.list([])
        makespans = manager.list([])
        average_completions = manager.list([])
        average_slowdowns = manager.list([])
        for i in range(n_episode):
            algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
                                    features_normalize_func=features_normalize_func)
            episode = Episode(machine_configs, jobs_configs, algorithm, None)
            algorithm.reward_giver.attach(episode.simulation)
            p = Process(target=multiprocessing_run,
                        args=(episode, trajectories, makespans, average_completions, average_slowdowns))

            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        agent.log('makespan', np.mean(makespans), agent.global_step)
        agent.log('average_completions', np.mean(average_completions), agent.global_step)
        agent.log('average_slowdowns', np.mean(average_slowdowns), agent.global_step)

        toc = time.time()

        # print(np.mean(makespans), toc - tic, np.mean(average_completions), np.mean(average_slowdowns))
        print(
            f"mean makespans: {np.mean(makespans)} | 'toc-tic: {toc - tic} | avg_completions: {np.mean(average_completions)} | avg_slowdowns: {np.mean(average_slowdowns)}")
        all_observations = []
        all_actions = []
        all_rewards = []
        for trajectory in trajectories:
            observations = []
            actions = []
            rewards = []
            for node in trajectory:
                observations.append(node.observation)
                actions.append(node.action)
                rewards.append(node.reward)

            all_observations.append(observations)
            all_actions.append(actions)
            all_rewards.append(rewards)

        all_q_s, all_advantages = agent.estimate_return(all_rewards)

        agent.update_parameters(all_observations, all_actions, all_advantages)

        if itr % save_chkpt_every == 0:
            save_train_info(agent, itr)

    agent.save()


def eval_algo_deep_js():
    # chkpt_path = "playground/Non_DAG/launch_scripts/agents/Makespan-Brain-m5/checkpoint"
    # chkpt_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/Makespan-Brain-m5/model.ckpt-8"
    # chkpt_path="/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/checkpoint-180"
    # chkpt_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/chkpt_180.pkl-7"
    chkpt_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/chkpt_120.pkl-5"
    agent = Agent(name, brain, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
                  model_save_path='%s/model.ckpt' % model_dir, restore_path=chkpt_path)
    tic = time.time()
    print("********** Eval DeepJS Agent ************")
    processes = []

    manager = Manager()
    trajectories = manager.list([])
    makespans = manager.list([])
    average_completions = manager.list([])
    average_slowdowns = manager.list([])
    # for i in range(n_episode):
    algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
                            features_normalize_func=features_normalize_func)
    episode = Episode(machine_configs, jobs_configs, algorithm, None)
    algorithm.reward_giver.attach(episode.simulation)
    p = Process(target=multiprocessing_run,
                args=(episode, trajectories, makespans, average_completions, average_slowdowns))

    processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    agent.log('makespan', np.mean(makespans), agent.global_step)
    agent.log('average_completions', np.mean(average_completions), agent.global_step)
    agent.log('average_slowdowns', np.mean(average_slowdowns), agent.global_step)

    toc = time.time()

    print(np.mean(makespans), toc - tic, np.mean(average_completions), np.mean(average_slowdowns))
    print(f"before makespans ({makespans})")
    print(f"mean makespan: {np.mean(makespans)}")
    print(f"toc - tic: {toc - tic}")
    print(f"average_completions: {np.mean(average_completions)}")
    print(f"average slowdowns: {np.mean(average_slowdowns)}")


def run_all_algo():
    algo_random()
    algo_first_fit()
    algo_tetris()
    # train_algo_deep_js()


if __name__ == '__main__':
    # run_all_algo()
    # algo_deep_js()
    # eval_algo_deep_js()
    # set_path()  # for running on command line
    train_algo_deep_js()
    # run_all_algo()

# DeepJS
# before makespans ([654])
# mean makespan: 654.0
# toc - tic: 2.6973419189453125
# average_completions: 144.16438373244327
# average slowdowns: 2.796577900344812


# DeepJS chkpt-180
# 612.0 2.818004846572876 182.63750201201316 4.002450774266493
# before makespans ([612])
# mean makespan: 612.0
# toc - tic: 2.818004846572876
# average_completions: 182.63750201201316
# average slowdowns: 4.002450774266493

# DeepJS chkpt-120
# 663.0 2.9996590614318848 141.8202977109379 2.6516431444454556
# before makespans ([663])
# mean makespan: 663.0
# toc - tic: 2.9996590614318848
# average_completions: 141.8202977109379
# average slowdowns: 2.6516431444454556


# random, first fit, tetric
# 641 0.4405100345611572 138.22889986147553 3.1337322343504233
# 680 0.44473695755004883 62.05685685072286 1.4292964198242561
# 689 1.2516517639160156 85.23965254964759 1.975667948996756
