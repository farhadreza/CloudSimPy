import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from playground.Non_DAG.utils.common_tokens import *


def load_df(csv_path=None, ignore_idx=False):
    if csv_path:
        if ignore_idx:
            df = pd.read_csv(csv_path, index_col=[0])
        else:
            df = pd.read_csv(csv_path)
        return df


def orgnize_RAC_train_stats():
    # hist_deepjs_train
    p0 = "experiments/data/train_stats_raw/RAC/hist_deepjs_train_RAC.csv"
    # p1 = "experiments/data/train_stats_raw/RAC/hist_deepjs_train_RAC_60.csv"
    p2 = "experiments/data/train_stats_raw/RAC/hist_deepjs_train_RAC_61.csv"
    p3 = "experiments/data/train_stats_raw/RAC/hist_deepjs_train_RAC_81.csv"
    p4 = "experiments/data/train_stats_raw/RAC/hist_deepjs_train_RAC_161.csv"
    plist = [p0, p2, p3, p4]
    save_path = get_train_stats_path(file_type="deepjs", reward_type=RAC)
    orgnize_train_stats(plist=plist, save_path=save_path)
    # reward stats
    p0 = "experiments/data/train_stats_raw/RAC/hist_reward_RAC.csv"
    # p1 = "experiments/data/train_stats_raw/RAC/hist_reward_RAC_60.csv"
    p2 = "experiments/data/train_stats_raw/RAC/hist_reward_RAC_61.csv"
    p3 = "experiments/data/train_stats_raw/RAC/hist_reward_RAC_81.csv"
    p4 = "experiments/data/train_stats_raw/RAC/hist_reward_RAC_161.csv"
    plist = [p0, p2, p3, p4]
    save_path = get_train_stats_path(file_type="reward", reward_type=RAC)
    orgnize_train_stats(plist=plist, save_path=save_path)

    # other algo stats during the same training
    p0 = "experiments/data/train_stats_raw/RAC/hist_RAC.csv"
    # p1 = "experiments/data/train_stats_raw/RAC/hist_RAC_60.csv"
    p2 = "experiments/data/train_stats_raw/RAC/hist_RAC_61.csv"
    p3 = "experiments/data/train_stats_raw/RAC/hist_RAC_81.csv"
    p4 = "experiments/data/train_stats_raw/RAC/hist_RAC_161.csv"

    plist = [p0, p2, p3, p4]
    save_path = get_train_stats_path(file_type="other_algo", reward_type=RAC)
    orgnize_train_stats(plist=plist, save_path=save_path)

    # len of total rows: 2000
    # len of total rows: 24000
    # len of total rows: 200


def orgnize_RAS_train_stats():
    # hist_deepjs_train
    p0 = "experiments/data/train_stats_raw/RAS/hist_deepjs_train_RAS.csv"
    p1 = "experiments/data/train_stats_raw/RAS/hist_deepjs_train_RAS_50.csv"
    p2 = "experiments/data/train_stats_raw/RAS/hist_deepjs_train_RAS_51.csv"
    p3 = "experiments/data/train_stats_raw/RAS/hist_deepjs_train_RAS_101.csv"
    p4 = "experiments/data/train_stats_raw/RAS/hist_deepjs_train_RAS_191.csv"
    plist = [p0, p1, p2, p3, p4]
    save_path = get_train_stats_path(file_type="deepjs", reward_type=RAS)
    orgnize_train_stats(plist=plist, save_path=save_path)
    # reward stats
    p0 = "experiments/data/train_stats_raw/RAS/hist_reward_RAS.csv"
    p1 = "experiments/data/train_stats_raw/RAS/hist_reward_RAS_50.csv"
    p2 = "experiments/data/train_stats_raw/RAS/hist_reward_RAS_51.csv"
    p3 = "experiments/data/train_stats_raw/RAS/hist_reward_RAS_101.csv"
    p4 = "experiments/data/train_stats_raw/RAS/hist_reward_RAS_191.csv"
    plist = [p0, p1, p2, p3, p4]
    save_path = get_train_stats_path(file_type="reward", reward_type=RAS)
    orgnize_train_stats(plist=plist, save_path=save_path)

    # other algo stats during the same training
    p0 = "experiments/data/train_stats_raw/RAS/hist_RAS.csv"
    p1 = "experiments/data/train_stats_raw/RAS/hist_RAS_50.csv"
    p2 = "experiments/data/train_stats_raw/RAS/hist_RAS_51.csv"
    p3 = "experiments/data/train_stats_raw/RAS/hist_RAS_101.csv"
    p4 = "experiments/data/train_stats_raw/RAS/hist_RAS_191.csv"
    plist = [p0, p1, p2, p3, p4]
    save_path = get_train_stats_path(file_type="other_algo", reward_type=RAS)
    orgnize_train_stats(plist=plist, save_path=save_path)
    # len of total rows: 2210
    # len of total rows: 26520
    # len of total rows: 221


def orgnize_RAM_train_stats():
    # hist_deepjs_train
    p0 = "experiments/data/train_stats_raw/RAM/hist_deepjs_train_RAM.csv"
    # p1 = "experiments/data/train_stats_raw/RAS/hist_deepjs_train_RAS_50.csv"

    plist = [p0]
    save_path = get_train_stats_path(file_type="deepjs", reward_type=RAM)
    orgnize_train_stats(plist=plist, save_path=save_path)
    # reward stats
    p0 = "experiments/data/train_stats_raw/RAM/hist_reward_RAM.csv"
    # p1 = "experiments/data/train_stats_raw/RAS/hist_reward_RAS_50.csv"

    plist = [p0]
    save_path = get_train_stats_path(file_type="reward", reward_type=RAM)
    orgnize_train_stats(plist=plist, save_path=save_path)

    # other algo stats during the same training
    p0 = "experiments/data/train_stats_raw/RAM/hist_RAM.csv"
    # p1 = "experiments/data/train_stats_raw/RAS/hist_RAS_50.csv"

    plist = [p0]
    save_path = get_train_stats_path(file_type="other_algo", reward_type=RAM)
    orgnize_train_stats(plist=plist, save_path=save_path)
    # len of total rows: 1910
    # len of total rows: 22920
    # len of total rows: 191


def orgnize_Mix_ACAS_train_stats():
    # hist_deepjs_train
    p0 = "experiments/data/train_stats_raw/MIX_RAC_RAS/hist_deepjs_train_MIX_AC_AS_0.csv"
    p1 = "experiments/data/train_stats_raw/MIX_RAC_RAS/hist_deepjs_train_MIX_AC_AS_71.csv"

    plist = [p0, p1]
    save_path = get_train_stats_path(file_type="deepjs", reward_type=MIX_AC_AS)
    orgnize_train_stats(plist=plist, save_path=save_path)
    # reward stats
    p0 = "experiments/data/train_stats_raw/MIX_RAC_RAS/hist_reward_MIX_AC_AS_0.csv"
    p1 = "experiments/data/train_stats_raw/MIX_RAC_RAS/hist_reward_MIX_AC_AS_71.csv"
    plist = [p0, p1]
    save_path = get_train_stats_path(file_type="reward", reward_type=MIX_AC_AS)
    orgnize_train_stats(plist=plist, save_path=save_path)

    # other algo stats during the same training
    p0 = "experiments/data/train_stats_raw/MIX_RAC_RAS/hist_MIX_AC_AS_0.csv"
    p1 = "experiments/data/train_stats_raw/MIX_RAC_RAS/hist_MIX_AC_AS_71.csv"

    plist = [p0, p1]
    save_path = get_train_stats_path(file_type="other_algo", reward_type=MIX_AC_AS)
    orgnize_train_stats(plist=plist, save_path=save_path)
    # len of total rows: 1910
    # len of total rows: 22920
    # len of total rows: 191


def orgnize_train_stats(plist, save_path=""):
    df_list = []
    if len(plist) == 1:
        df_all = load_df(plist[0])
    else:
        for fpath in plist:
            df = load_df(fpath)
            df_list.append(df)
        df_all = pd.concat(df_list)
    print(f"len of total rows: {len(df_all)}")
    df_all.to_csv(save_path, index=False)


def collect_orgnize_all_train_stats():
    """collect and put all train stats into one file
    """
    # orgnize_RAS_train_stats()
    # orgnize_RAC_train_stats()
    orgnize_RAM_train_stats()
