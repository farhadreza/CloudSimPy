import os
import time
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager
import sys

# sys.path.append('..')
sys.path.append('.')

from core.machine import MachineConfig
from playground.Non_DAG.algorithm.random_algorithm import RandomAlgorithm
from playground.Non_DAG.algorithm.tetris import Tetris
from playground.Non_DAG.algorithm.first_fit import FirstFitAlgorithm
from playground.Non_DAG.algorithm.DeepJS.DRL import RLAlgorithm
from playground.Non_DAG.algorithm.DeepJS.agent import Agent
from playground.Non_DAG.algorithm.DeepJS.brain import Brain, MyBrain

from playground.Non_DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver, AverageCompletionRewardGiver, \
    AverageSlowDownRewardGiver, AverageMix_RAC_RAS

from playground.Non_DAG.utils.csv_reader import CSVReader
from playground.Non_DAG.utils.feature_functions import features_extract_func, features_normalize_func
from playground.Non_DAG.utils.tools import multiprocessing_run, average_completion, average_slowdown
from playground.Non_DAG.utils.episode import Episode
import pandas as pd
from collections import defaultdict

from playground.Non_DAG.utils.common_tokens import *

os.environ['CUDA_VISIBLE_DEVICES'] = ''

np.random.seed(41)
tf.random.set_random_seed(41)
# ************************ Parameters Setting Start ************************
machines_number = 5
n_job_chunk = 200
eval_job_chunk = 521
jobs_len = 10
n_iter = 10
n_episode = 12

# n_iter = 200
# n_iter = 2
n_episode = 12
jobs_csv = 'playground/Non_DAG/jobs_files/jobs.csv'
# jobs_csv = '../jobs_files/jobs_2017.csv'

# brain = Brain(6)
brain = MyBrain(6)
# reward_giver = MakespanRewardGiver(-1)
reward_giver = AverageMix_RAC_RAS()
# reward_giver = AverageCompletionRewardGiver()
curr_reward_signal_name = "MIX_ACAS_MyBrain60"

features_extract_func = features_extract_func
features_normalize_func = features_normalize_func

name = '%s-%s-m%d' % (reward_giver.name, brain.name, machines_number)
model_dir = './agents/%s' % name

# train_info_dir = './agents/training/avgCompletionReward'
train_info_dir = './agents/train80/avgMakespan'
eval_info_dir = "experiments/data/eval/temp_eval"

trained_agent_path = "curr_agents/MIX_ACAS/brain_MIX_AC_AS_MyBrain_60.pkl"
# train_info_dir = "/content/drive/MyDrive/GoogleDrive/MyRepo/"
# ************************ Parameters Setting End ************************

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

# agent = Agent(name, brain, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
#               model_save_path='%s/model.ckpt' % model_dir)
agent = Agent(name, brain, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
              model_save_path='%s/model.ckpt' % train_info_dir,restore_path=trained_agent_path)

machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
single_jobs_configs = csv_reader.generate(0, jobs_len)
hist = defaultdict(list)
hist_eval = defaultdict(list)


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
def save_train_info(agent: Agent, itr: int, reward_type=curr_reward_signal_name):
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


def save_eval_info(reward_type=curr_reward_signal_name):
    if not os.path.exists(eval_info_dir):
        os.makedirs(eval_info_dir, exist_ok=True)
    hist_name = f"hist_{reward_type}_.csv"
    hist_path = os.path.join(eval_info_dir, hist_name)
    df = pd.DataFrame(hist_eval)
    df.to_csv(hist_path)
    print(f"save hist_eval to: {hist_path}")


# def add_result_to_hist(algo_type, env_now, toctic, avg_completion, avg_slowdown):
#     hist[algo_type + "_env_now"].append(env_now)
#     hist[algo_type + "_tictoc"].append(toctic)
#     hist[algo_type + "_avg_completions"].append(avg_completion)
#     hist[algo_type + "_avg_slowdowns"].append(avg_slowdown)
#
#
# def algorithm_random(print_stats=False):
#     tic = time.time()
#     algorithm = RandomAlgorithm()
#     episode = Episode(machine_configs, single_jobs_configs, algorithm, None)
#     episode.run()
#     if print_stats:
#         print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
#         print(
#             f"episode_env_now: {episode.env.now} | 'toc-tic: {time.time() - tic} | avg_completions: {average_completion(episode)} | avg_slowdowns: {average_slowdown(episode)}")
#     add_result_to_hist(algo_type="random", env_now=episode.env.now, toctic=time.time() - tic,
#                        avg_completion=average_completion(episode), avg_slowdown=average_slowdown(episode))
#
#
# def algorithm_first_fit(print_stats=False):
#     tic = time.time()
#     algorithm = FirstFitAlgorithm()
#     episode = Episode(machine_configs, single_jobs_configs, algorithm, None)
#     episode.run()
#     if print_stats:
#         print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
#         print(
#             f"episode_env_now: {episode.env.now} | 'toc-tic: {time.time() - tic} | avg_completions: {average_completion(episode)} | avg_slowdowns: {average_slowdown(episode)}")
#     add_result_to_hist(algo_type="first_fit", env_now=episode.env.now, toctic=time.time() - tic,
#                        avg_completion=average_completion(episode), avg_slowdown=average_slowdown(episode))
#
#
# def algorithm_tetris(print_stats=False):
#     tic = time.time()
#     algorithm = Tetris()
#     episode = Episode(machine_configs, single_jobs_configs, algorithm, None)
#     episode.run()
#     if print_stats:
#         print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
#         print(
#             f"episode_env_now: {episode.env.now} | 'toc-tic: {time.time() - tic} | avg_completions: {average_completion(episode)} | avg_slowdowns: {average_slowdown(episode)}")
#     add_result_to_hist(algo_type="tetris", env_now=episode.env.now, toctic=time.time() - tic,
#                        avg_completion=average_completion(episode), avg_slowdown=average_slowdown(episode))


save_chkpt_every = 10


def add_hist(name="", value=None):
    hist[name].append(value)


def add_train_stats_to_hist(algo_type, tictime, env_now, global_step, avg_compl, avg_mkspan, avg_slowd,
                            is_other_algo=False):
    hist[algo_type + "_" + tictoc].append(tictime)
    hist[algo_type + "_" + avg_makespans].append(avg_mkspan)
    hist[algo_type + "_" + env_now].append(env_now)
    hist[algo_type + "_" + globalstep].append(global_step)
    hist[algo_type + "-" + avg_completions].append(avg_compl)
    hist[algo_type + "_" + avg_slowdowns].append(avg_slowd)


def train_DeepJS_data200():
    for job_chunk in range(n_job_chunk):
        jobs_configs = csv_reader.generate(job_chunk * jobs_len, jobs_len, hist=hist)

        tic = time.time()
        algorithm = RandomAlgorithm()
        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        episode.run()
        # add_train_stats_to_hist(algo_random,)
        hist["random_tictoc"].append(time.time() - tic)
        hist["random_avg_makespans"].append(episode.env.now)
        hist["makespan-random_env_now"].append(episode.env.now)
        hist["random_global_step"].append(agent.global_step)
        hist["random_avg_completions"].append(average_completion(episode))
        hist["random_avg_slowdowns"].append(average_slowdown(episode))

        print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
        agent.log('makespan-random', episode.env.now, agent.global_step)

        tic = time.time()
        algorithm = FirstFitAlgorithm()
        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        episode.run()
        hist["first_fit_tictoc"].append(time.time() - tic)
        hist["first_fit_avg_makespans"].append(episode.env.now)
        hist["makespan-ff_env_now"].append(episode.env.now)
        hist["first_fit_global_step"].append(agent.global_step)
        hist["first_fit_avg_completions"].append(average_completion(episode))
        hist["first_fit_avg_slowdowns"].append(average_slowdown(episode))

        print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
        agent.log('makespan-ff', episode.env.now, agent.global_step)
        # hist["makespan-ff_env_now"].append(episode.env.now)
        # hist["makespan-ff_global_step"].append(agent.global_step)

        tic = time.time()
        algorithm = Tetris()
        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        episode.run()
        hist["tetris_tictoc"].append(time.time() - tic)
        hist["tetris_avg_makespans"].append(episode.env.now)
        hist["makespan-tetris_env_now"].append(episode.env.now)
        hist["tetris_global_step"].append(agent.global_step)
        hist["tetris_avg_completions"].append(average_completion(episode))
        hist["tetris_avg_slowdowns"].append(average_slowdown(episode))

        print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
        agent.log('makespan-tetris', episode.env.now, agent.global_step)

        for itr in range(n_iter):
            tic = time.time()
            print("********** Iteration %i ************" % itr)
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
            # agent.log('average_completions', np.mean(average_completions), agent.global_step)
            # agent.log('average_slowdowns', np.mean(average_slowdowns), agent.global_step)
            toc = time.time()
            hist[curr_reward_signal_name + "_avg_makespans"].append(np.mean(makespans))
            hist[curr_reward_signal_name + "_avg_completions"].append(np.mean(average_completions))
            hist[curr_reward_signal_name + "_avg_slowdowns"].append(np.mean(average_slowdowns))
            hist[curr_reward_signal_name + "_global_step"].append(agent.global_step)
            hist[curr_reward_signal_name + "_tictoc"].append(toc - tic)
            print(np.mean(makespans), toc - tic, np.mean(average_completions), np.mean(average_slowdowns))

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

                add_hist(curr_reward_signal_name + '_total_rewards', np.sum(rewards))
                add_hist(curr_reward_signal_name + "_avg_rewards", np.mean(rewards))
            all_q_s, all_advantages = agent.estimate_return(all_rewards)

            agent.update_parameters(all_observations, all_actions, all_advantages)
        if job_chunk % save_chkpt_every == 0 or job_chunk == 200:
            save_train_info(agent, job_chunk)
        agent.save()


def eval_other_algos_data200(algo_name="", algorithm=None, save_dir=None, print_progress=True):
    start_job_no = 201
    report_every = 10
    eval_dict = defaultdict(list)
    for job_chunk in range(start_job_no, eval_job_chunk):
        if print_progress and job_chunk % report_every == 0:
            print(f"************* job_chunk: {job_chunk} ***************")
        jobs_configs = csv_reader.generate(job_chunk * jobs_len, jobs_len, print_stats=False)

        tic = time.time()
        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        episode.run()
        # add_train_stats_to_hist(algo_random,)
        eval_dict[algo_name + "_tictoc"].append(time.time() - tic)
        eval_dict[algo_name + "_avg_makespans"].append(episode.env.now)
        eval_dict[algo_name + "_env_now"].append(episode.env.now)
        # eval_dict["random_global_step"].append(agent.global_step)
        eval_dict[algo_name + "_avg_completions"].append(average_completion(episode))
        eval_dict[algo_name + "_avg_slowdowns"].append(average_slowdown(episode))
    if save_dir:
        save_name = f"hist_{algo_name}.csv"
        save_path = os.path.join(save_dir, save_name)
        df_eval = pd.DataFrame(eval_dict)
        df_eval.to_csv(save_path)
        print(f"saved {algo_name} algo to: {save_path}")


def eval_all_other_algos():
    """
    evaluate all other algorithms other than DeepJS
    save the result to eval_info_dir
    :return:
    """
    save_dir = eval_info_dir

    algorithm = RandomAlgorithm()
    # eval_other_algos_data200(algo_name=algo_random, algorithm=algorithm, save_dir=save_dir)
    algorithm = FirstFitAlgorithm()
    # eval_other_algos_data200(algo_name=algo_first_fit, algorithm=algorithm, save_dir=save_dir)
    algorithm = Tetris()
    eval_other_algos_data200(algo_name=algo_tetris, algorithm=algorithm, save_dir=save_dir)


def eval_DeepJS_data200(reward_type=curr_reward_signal_name):
    start_job_no = 201
    algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
                            features_normalize_func=features_normalize_func)
    eval_dict = defaultdict(list)
    print_progress = True
    for job_chunk in range(start_job_no, eval_job_chunk):
        if print_progress:
            print(f"************* job_chunk: {job_chunk} ***************")
        jobs_configs = csv_reader.generate(job_chunk * jobs_len, jobs_len, hist=hist)
        #
        # tic = time.time()
        # algorithm = RandomAlgorithm()
        # episode = Episode(machine_configs, jobs_configs, algorithm, None)
        # episode.run()
        # # add_train_stats_to_hist(algo_random,)
        # hist["random_tictoc"].append(time.time() - tic)
        # hist["random_avg_makespans"].append(episode.env.now)
        # hist["makespan-random_env_now"].append(episode.env.now)
        # hist["random_global_step"].append(agent.global_step)
        # hist["random_avg_completions"].append(average_completion(episode))
        # hist["random_avg_slowdowns"].append(average_slowdown(episode))
        #
        # print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
        # agent.log('makespan-random', episode.env.now, agent.global_step)
        #
        # tic = time.time()
        # algorithm = FirstFitAlgorithm()
        # episode = Episode(machine_configs, jobs_configs, algorithm, None)
        # episode.run()
        # hist["first_fit_tictoc"].append(time.time() - tic)
        # hist["first_fit_avg_makespans"].append(episode.env.now)
        # hist["makespan-ff_env_now"].append(episode.env.now)
        # hist["first_fit_global_step"].append(agent.global_step)
        # hist["first_fit_avg_completions"].append(average_completion(episode))
        # hist["first_fit_avg_slowdowns"].append(average_slowdown(episode))
        #
        # print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
        # agent.log('makespan-ff', episode.env.now, agent.global_step)
        # # hist["makespan-ff_env_now"].append(episode.env.now)
        # # hist["makespan-ff_global_step"].append(agent.global_step)
        #
        # tic = time.time()
        # algorithm = Tetris()
        # episode = Episode(machine_configs, jobs_configs, algorithm, None)
        # episode.run()
        # hist["tetris_tictoc"].append(time.time() - tic)
        # hist["tetris_avg_makespans"].append(episode.env.now)
        # hist["makespan-tetris_env_now"].append(episode.env.now)
        # hist["tetris_global_step"].append(agent.global_step)
        # hist["tetris_avg_completions"].append(average_completion(episode))
        # hist["tetris_avg_slowdowns"].append(average_slowdown(episode))
        #
        # print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))
        # agent.log('makespan-tetris', episode.env.now, agent.global_step)

        tic = time.time()
        # manager = Manager()
        # trajectories = manager.list([])
        # makespans = manager.list([])
        # average_completions = manager.list([])
        # average_slowdowns = manager.list([])
        makespans = []
        average_completions = []
        average_slowdowns = []

        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        algorithm.reward_giver.attach(episode.simulation)
        # multiprocessing_run(episode, trajectories, makespans, average_completions, average_slowdowns)

        episode.run()
        makespans.append(episode.simulation.env.now)
        average_completions.append(average_completion(episode))
        average_slowdowns.append(average_slowdown(episode))
        # p = Process(target=multiprocessing_run,
        #             args=(episode, trajectories, makespans, average_completions, average_slowdowns))
        #
        # processes.append(p)
        #
        # for p in processes:
        #     p.start()
        #
        # for p in processes:
        #     p.join()

        # agent.log('makespan', np.mean(makespans), agent.global_step)
        # agent.log('average_completions', np.mean(average_completions), agent.global_step)
        # agent.log('average_slowdowns', np.mean(average_slowdowns), agent.global_step)
        toc = time.time()
        eval_dict[curr_reward_signal_name + "_avg_makespans"].append(np.mean(makespans))
        eval_dict[curr_reward_signal_name + "_avg_completions"].append(np.mean(average_completions))
        eval_dict[curr_reward_signal_name + "_avg_slowdowns"].append(np.mean(average_slowdowns))
        # eval_dict[curr_reward_signal_name + "_global_step"].append(agent.global_step)
        eval_dict[curr_reward_signal_name + "_tictoc"].append(toc - tic)
        # print(np.mean(makespans), toc - tic, np.mean(average_completions), np.mean(average_slowdowns))

    save_name = f"hist_{reward_type}.csv"
    save_path = os.path.join(eval_info_dir, save_name)
    df_eval = pd.DataFrame(eval_dict)
    df_eval.to_csv(save_path)


def exp_eval():
    """
    eval all algorithms for experiment
    :return:
    """
    # eval_all_other_algos()
    eval_DeepJS_data200()


#
# def train_algo_deep_js():
#     for itr in range(n_iter):
#         tic = time.time()
#         print("********** DeepJS Iteration %i ************" % itr)
#         processes = []
#
#         manager = Manager()
#         trajectories = manager.list([])
#         makespans = manager.list([])
#         average_completions = manager.list([])
#         average_slowdowns = manager.list([])
#         for i in range(n_episode):
#             algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
#                                     features_normalize_func=features_normalize_func)
#             episode = Episode(machine_configs, single_jobs_configs, algorithm, None)
#             algorithm.reward_giver.attach(episode.simulation)
#             p = Process(target=multiprocessing_run,
#                         args=(episode, trajectories, makespans, average_completions, average_slowdowns))
#
#             processes.append(p)
#
#         for p in processes:
#             p.start()
#
#         for p in processes:
#             p.join()
#
#         agent.log('makespan', np.mean(makespans), agent.global_step)
#         agent.log('average_completions', np.mean(average_completions), agent.global_step)
#         agent.log('average_slowdowns', np.mean(average_slowdowns), agent.global_step)
#
#         toc = time.time()
#
#         # print(np.mean(makespans), toc - tic, np.mean(average_completions), np.mean(average_slowdowns))
#         print(
#             f"mean makespans: {np.mean(makespans)} | 'toc-tic: {toc - tic} | avg_completions: {np.mean(average_completions)} | avg_slowdowns: {np.mean(average_slowdowns)}")
#         hist['avg_makespans'].append(np.mean(makespans))
#         hist['tictoc'].append(toc - tic)
#         hist['avg_completions'].append(np.mean(average_completions))
#         hist['avg_slowdowns'].append(np.mean(average_slowdowns))
#
#         all_observations = []
#         all_actions = []
#         all_rewards = []
#         for trajectory in trajectories:
#             observations = []
#             actions = []
#             rewards = []
#             for node in trajectory:
#                 observations.append(node.observation)
#                 actions.append(node.action)
#                 rewards.append(node.reward)
#
#             all_observations.append(observations)
#             all_actions.append(actions)
#             all_rewards.append(rewards)
#         add_hist('total_rewards', np.sum(rewards))
#         add_hist("avg_rewards", np.mean(rewards))
#
#         all_q_s, all_advantages = agent.estimate_return(all_rewards)
#
#         agent.update_parameters(all_observations, all_actions, all_advantages)
#
#         if itr % save_chkpt_every == 0 or itr == n_iter:
#             save_train_info(agent, itr)
#
#     agent.save()


def eval_algo_deep_js():
    # chkpt_path = "playground/Non_DAG/launch_scripts/agents/Makespan-Brain-m5/checkpoint"
    # chkpt_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/Makespan-Brain-m5/model.ckpt-8"
    # chkpt_path="/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/checkpoint-180"
    # chkpt_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/chkpt_180.pkl-7"
    # chkpt_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/chkpt_120.pkl-5"
    chkpt_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/avgCompletionReward/chkpt_180_mkspan.pkl-10"
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
    episode = Episode(machine_configs, single_jobs_configs, algorithm, None)
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


def run_other_algo():
    algo_random()
    algo_first_fit()
    algo_tetris()
    # train_algo_deep_js()
    save_to = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/launch_scripts/agents/training/otherAlgo/hist_other_algo.csv"
    df = pd.DataFrame(hist)
    df.to_csv(save_to)
    print(f"saved hist.")


def save_other_algo_hist_to_csv(save_dir=""):
    df_random = pd.DataFrame()
    df_tetris = pd.DataFrame()
    df_first_fit = pd.DataFrame()
    for algo_name, value in hist.items():
        if algo_name.startswith("random"):
            df_random[algo_name] = value
        elif algo_name.startswith("tetris"):
            df_tetris[algo_name] = value
        elif algo_name.startswith("first_fit"):
            df_first_fit[algo_name] = value
    random_path = os.path.join(save_dir, 'hist_random.csv')
    tetris_path = os.path.join(save_dir, "hist_tetris.csv")
    first_fit_path = os.path.join(save_dir, "hist_first_fit.csv")
    df_random.to_csv(random_path)
    df_tetris.to_csv(tetris_path)
    df_first_fit.to_csv(first_fit_path)


def run_other_algo_multiple_times():
    """
    run the scheduling algo multiple times to get a overrall performance
    :return:
    """
    num_times = 180
    for idx in range(num_times):
        print(f"running random algo")
        algo_random()
    for idx in range(num_times):
        print(f"running first fit algo")
        algo_first_fit()
    for idx in range(num_times):
        print(f"running tetris algo")
        algo_tetris()
    save_dir = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/renamed_reward_files"
    save_other_algo_hist_to_csv(save_dir=save_dir)
    print(f"saved hist.")


if __name__ == '__main__':
    # run_all_algo()
    # algo_deep_js()
    # eval_algo_deep_js()
    # set_path()  # for running on command line
    eval_DeepJS_data200()
    # eval_algo_deep_js()

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
