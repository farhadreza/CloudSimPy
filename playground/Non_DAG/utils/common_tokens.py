import os

avg_slowdowns = "avg_slowdowns"
avg_completions = "avg_completions"
avg_rewards = "avg_rewards"
avg_makespans = "avg_makespans"
globalstep = "global_step"
tictoc = "tictoc"
total_rewards = "total_rewards"
algo_random = "random"
algo_tetris = "tetris"
algo_first_fit = "first_fit"
other_algo = "other_algo"

exp_data_dir = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data"
job_file_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/jobs_files/jobs.csv"
fig_dir = f"{exp_data_dir}/fig"

RAS = "RAS"
RAM = "RAM"
RAC = "RAC"
MIX_AC_AS = "MIX_ACAS"

reward_avg_completions = RAC
reward_avg_makespans = RAM
reward_avg_slowdowns = RAS
reward_mix_acas = MIX_AC_AS
# reward_avg_completions = "reward_avg_completions"
# reward_avg_makespans = "reward_avg_makespans"
# reward_avg_slowdowns = "reward_avg_slowdowns"
all_cols = [avg_makespans, tictoc, avg_completions, avg_slowdowns, total_rewards, avg_rewards]
other_algo_list = [algo_random, algo_first_fit, algo_first_fit]


def get_exp_file_path(file_type="", is_original=False, prefix="hist_"):
    if is_original:  # for old data stats, don't use it
        fdir = os.path.join(exp_data_dir, "original_reward_files")
    else:
        # fdir = os.path.join(exp_data_dir, "renamed_reward_files")
        fdir = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/eval/raw"

    fpath = os.path.join(fdir, prefix + file_type + ".csv")
    return fpath
    # if is_original:
    #     if file_type == avg_completions:
    #         hist_avg_completions_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/original_reward_files/hist_avg_completions.csv"
    #         return hist_avg_completions_path
    #     elif file_type == avg_slowdowns:
    #         hist_avg_slowdowns = ""
    #         return hist_avg_slowdowns
    #     elif file_type == avg_makespans:
    #         hist_avg_makespans = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/original_reward_files/hist_avg_makespans.csv"
    #         return hist_avg_makespans
    #     elif file_type in [algo_random, algo_tetris, algo_first_fit]:
    #         hist_othr_algo = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/original_reward_files/hist_other_algo.csv"
    # else:
    #     if file_type == avg_completions:
    #         hist_avg_completions_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/original_reward_files/hist_avg_completions.csv"
    #         return hist_avg_completions_path
    #     elif file_type == avg_slowdowns:
    #         hist_avg_slowdowns = ""
    #         return hist_avg_slowdowns
    #     elif file_type == avg_makespans:
    #         hist_avg_makespans = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/original_reward_files/hist_avg_makespans.csv"
    #         return hist_avg_makespans


def get_train_stats_path(file_type="", reward_type="", prefix="hist_"):
    fdir = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/data/train_stats"
    fpath = os.path.join(fdir, prefix + reward_type + "_" + file_type + ".csv")
    return fpath
