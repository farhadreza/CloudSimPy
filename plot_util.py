import os, re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from playground.Non_DAG.utils.common_tokens import *

total_iters = 180


def load_df(csv_path=None, ignore_idx=False):
    if csv_path:
        if ignore_idx:
            df = pd.read_csv(csv_path, index_col=[0])
        else:
            df = pd.read_csv(csv_path)
        return df


def rename_df_col(df, prefix="", save_to=""):
    if not df is None:
        cols = df.columns.values.tolist()
        df_renamed = pd.DataFrame()
        for col in cols:
            if col and col != "Unnamed: 0":
                new_name = f"{prefix}{col}"
                df_renamed[new_name] = df[col]
            else:
                df_renamed[col] = df[col]
        if save_to:
            df_renamed.to_csv(save_to, index=False)


def hist_deepjs_avg():
    hist_rac_path = ""
    df = load_df(hist_rac_path)
    hist_rac = defaultdict(list)
    niters = 10
    total_len = len(hist_rac)
    start = 0
    cols = ["RAC_avg_makespans", "RAC_avg_completions", "RAC_avg_slowdowns"]
    for col in cols:
        for step in range(0, total_len, niters):
            end = start + niters
            ls = df.loc[start:end, col]
            hist_rac[col].append(sum(ls) / len(ls))

    df_avg = pd.DataFrame(hist_rac)
    save_to = "curr_agents/hist_rac_deepjs40.csv"
    df_avg.to_csv(save_to)


def rename_add_reward_type():
    """
    rename column name for plotting later
    only need to do it once
    :return:
    """
    is_original = True
    hist_completions = get_exp_file_path(file_type=avg_completions, is_original=is_original)
    hist_makespans = get_exp_file_path(file_type=avg_makespans, is_original=is_original)
    hist_slowdowns = get_exp_file_path(file_type=avg_slowdowns, is_original=is_original)
    df_completions = load_df(hist_completions)
    df_makespans = load_df(hist_makespans)
    df_slowdowns = load_df(hist_slowdowns)
    # rename_df_col(df_completions, prefix=reward_avg_completions + "_",
    #               save_to=get_exp_file_path(file_type=avg_completions, is_original=False))
    # rename_df_col(df_makespans, prefix=reward_avg_makespans + "_",
    #               save_to=get_exp_file_path(file_type=avg_makespans, is_original=False))
    rename_df_col(df_slowdowns, prefix=reward_avg_slowdowns + "_",
                  save_to=get_exp_file_path(file_type=avg_slowdowns, is_original=False))
    print(f"renamed columns.")


def plot_lines_from_df(df, cols=None, title=None, y_label=None):
    # if not cols is None:
    #     cols = df.columns.values.tolist()
    fig = plt.figure()
    num_sub_plots = 1
    gs = fig.add_gridspec(num_sub_plots, hspace=0)
    # axs = gs.subplots(sharex=True, sharey=True)
    axs = gs.subplots()
    # fig.suptitle("this is the super title")

    figsize = (8, 8)
    if num_sub_plots == 1:
        # df.plot(marker='.', figsize=figsize, ax=axs, color=get_column_color(df1))
        df.plot(marker='.', figsize=figsize, ax=axs)
    if title:
        axs.set_title(title)
    if y_label:
        axs.set_ylabel(y_label)
    # for ax in axs:
    #     ax.label_outer()
    # plt.savefig(save_to)
    plt.show()


def plot_all_job_stats_from_df(df, cols=None, show_y_lable=True, save_to=None):
    if cols is None:
        return
    fig = plt.figure()
    num_sub_plots = len(cols)
    gs = fig.add_gridspec(1, num_sub_plots, hspace=0)
    # axs = gs.subplots(sharex=True, sharey=True)
    axs = gs.subplots(sharex=True, sharey=False)
    # fig.suptitle("this is the super title")

    figsize = (24, 8)
    if num_sub_plots == 1:
        # df.plot(marker='.', figsize=figsize, ax=axs, color=get_column_color(df1))
        df.plot(figsize=figsize, ax=axs)
    else:
        for idx, col in enumerate(cols):
            df[col].plot(figsize=figsize, ax=axs[idx])
            if show_y_lable:
                lable = col.capitalize() if col != "instances_num" else "Number of Instances"
                axs[idx].set_ylabel(lable)
    # for ax in axs:
    #     ax.label_outer()
    if save_to:
        fpath = os.path.join(save_to, "job_stats_all.png")
        plt.savefig(fpath)
    else:
        plt.show()


def plot_individual_job_stats_from_df(df, cols=None, show_y_lable=True, save_to=None):
    if cols is None:
        return
    fig = plt.figure()
    num_sub_plots = 1
    gs = fig.add_gridspec(num_sub_plots, hspace=0)
    # axs = gs.subplots(sharex=True, sharey=True)
    axs = gs.subplots()
    # fig.suptitle("this is the super title")

    figsize = (8, 8)
    if num_sub_plots == 1:
        # df.plot(marker='.', figsize=figsize, ax=axs, color=get_column_color(df1))
        df[cols].plot(figsize=figsize, ax=axs)
        if show_y_lable:
            lable = cols.capitalize() if cols != "instances_num" else "Number of Instances"
            axs.set_ylabel(lable)
    if save_to:
        fpath = os.path.join(save_to, "job_" + cols + ".png")
        plt.savefig(fpath)
    else:
        plt.show()


def plot_stats_for_reward_signal(reward_type="", rename_col=True, use_cols=None, title=None, y_label=None):
    csv_path = get_exp_file_path(file_type=reward_type)
    df_stats = load_df(csv_path)
    # cols = ["reward_" + reward_type + "_" + avg_makespans, "reward_" + reward_type + "_" + avg_slowdowns,
    #         "reward_" + reward_type + "_" + avg_completions]
    cols = [avg_makespans, avg_slowdowns, avg_completions]
    if use_cols:
        cols = use_cols
    if rename_col:
        df = pd.DataFrame()
        for col in cols:
            df[col] = df_stats["reward_" + reward_type + "_" + col]
        df_stats = df.copy()
    plot_lines_from_df(df_stats[cols], title=title, y_label=y_label)


def plot_reward_avg_completions():
    plot_stats_for_reward_signal(reward_type=avg_completions, rename_col=True)


def plot_reward_avg_makespans():
    plot_stats_for_reward_signal(reward_type=avg_makespans, rename_col=True)


def plot_reward_avg_slowdowns():
    plot_stats_for_reward_signal(reward_type=avg_slowdowns, rename_col=True)


def plot_training_stats_for_reward_all():
    use_cols = [avg_completions]
    plot_stats_for_reward_signal(reward_type=avg_completions, rename_col=True, use_cols=use_cols,
                                 title=f"reward_{avg_completions}")
    plot_stats_for_reward_signal(reward_type=avg_makespans, rename_col=True, use_cols=use_cols,
                                 title=f"reward_{avg_makespans}")
    plot_stats_for_reward_signal(reward_type=avg_slowdowns, rename_col=True, use_cols=use_cols,
                                 title=f"reward_{avg_slowdowns}")


def plot_reward_data(dfs, cols=None):
    """
    plot avg_completion, avg_makespan, or avg_slowdown for different reward signals
    dfs contains a list of tuples, each tuple is of format (reward_type, df)
    this method plots one stat for multiple reward_types
    :param dfs:
    :param cols:
    :return:
    """
    if dfs:
        df_stats = pd.DataFrame()
        for reward_type, df in dfs:
            df_cols = df.columns.values.tolist()
            for col, curr_col in zip(cols, df_cols):
                if col and curr_col.endswith(col):
                    df_stats[reward_type] = df[curr_col]
        plot_lines_from_df(df_stats)


def plot_single_stat_for_all_reward_signals(dfs, col_name=None, y_label=None, fig_dir=None, figsize=(10, 10),
                                            excludes=None, x_lable=None):
    if dfs and col_name:
        df_stats = pd.DataFrame()
        for reward_type, df in dfs:
            cols = df.columns.values.tolist()
            for col in cols:
                if reward_type in other_algo_list and col_name == avg_makespans:
                    col_name = "env_now"
                if col and col.endswith(col_name):
                    if excludes and col in excludes:  # don't plot certain stats if specified
                        continue
                    df_stats[reward_type] = df[col]
        fig = plt.figure()
        num_sub_plots = 1
        gs = fig.add_gridspec(num_sub_plots, hspace=0)
        # axs = gs.subplots(sharex=True, sharey=True)
        axs = gs.subplots()
        # fig.suptitle("this is the super title")

        if num_sub_plots == 1:
            # df.plot(marker='.', figsize=figsize, ax=axs, color=get_column_color(df1))
            df_stats.plot(marker='.', figsize=figsize, ax=axs)
        if y_label:
            axs.set_ylabel(y_label)
        if x_lable:
            axs.set_xlabel(xlabel=x_lable)
        # for ax in axs:
        #     ax.label_outer()
        if fig_dir:
            figpath = os.path.join(fig_dir, "stats_" + col_name + ".png")
            plt.savefig(figpath)
        else:
            plt.show()


def get_coloumn_name_ends_with(cols, suffix=None):
    if cols and suffix:
        name = [col for col in cols if col.endswith(suffix)]
        if name:
            # temp method for finding a matching name, names are unique so only one should be in the lsit
            return name[0]


def plot_stat_difference_for_all_reward_signals(dfs, compare_stat=None, reward_signal_name="", y_label=None,
                                                fig_dir=None,
                                                figsize=(10, 10), excludes=None, df_compare_to=None,
                                                compare_to_col_name="", suffix="", x_lable=None):
    """
    this loops through each file and compare a specific stat (specified by the compare_stat) to the same
    stat trained for all other reward signals
    :param dfs:
    :param compare_stat:
    :param reward_signal_name:
    :param y_label:
    :param fig_dir:
    :param figsize:
    :param excludes:
    :param df_compare_to:
    :return:
    """
    if dfs and compare_stat and not df_compare_to is None:
        df_stats = pd.DataFrame()
        for reward_type, df in dfs:
            cols = df.columns.values.tolist()
            for col in cols:
                if reward_type in other_algo_list and compare_stat == avg_makespans:
                    compare_stat = "env_now"
                if col and col.endswith(compare_stat):
                    if excludes and col in excludes:  # don't plot certain stats if specified
                        continue
                    # others minus the current stats to get the difference between performance
                    if reward_type in other_algo_list and col == avg_makespans:  # used a different name for other algo, so need to accomadate it
                        # col_name=reward_type + "_env_now"
                        df_stats[reward_type] = df[reward_type + "_env_now"] - df_compare_to[compare_to_col_name]
                    else:
                        curr_col = get_coloumn_name_ends_with(cols, suffix=compare_stat)
                        df_stats[reward_type] = df[curr_col] - df_compare_to[compare_to_col_name]
        fig = plt.figure()
        num_sub_plots = 1
        gs = fig.add_gridspec(num_sub_plots, hspace=0)
        # axs = gs.subplots(sharex=True, sharey=True)
        axs = gs.subplots()
        # fig.suptitle("this is the super title")

        if num_sub_plots == 1:
            # df.plot(marker='.', figsize=figsize, ax=axs, color=get_column_color(df1))
            df_stats.plot(marker='.', figsize=figsize, ax=axs)
        if y_label:
            axs.set_ylabel(y_label)
        if x_lable:
            axs.set_xlabel(xlabel=x_lable)
        # for ax in axs:
        #     ax.label_outer()
        if fig_dir:
            fname = reward_signal_name if reward_signal_name else compare_stat
            figpath = os.path.join(fig_dir, "stat_diff_" + fname + suffix + ".png")
            plt.savefig(figpath)
        else:
            plt.show()


# @exp_fig
def exp_results_by_reward_all_plots():
    """
    this plots avg_completions, avg_makespans, avg_slowdowns for all reward signals
    :return:
    """
    # if fig dir is specified, the figs will be save to the dir
    # fig_dir = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/figs/by_reward/all_stat"
    df_completion = load_df(get_exp_file_path(file_type=avg_completions))
    df_makespan = load_df(get_exp_file_path(file_type=avg_makespans))
    df_slowdown = load_df(get_exp_file_path(file_type=avg_slowdowns))
    # df_others = load_df(get_exp_file_path(file_type=other_algo))
    df_tetris = load_df(get_exp_file_path(file_type=algo_tetris))
    df_random = load_df(get_exp_file_path(file_type=algo_random))
    df_first_fit = load_df(get_exp_file_path(file_type=algo_first_fit))
    dfs = [(RAC, df_completion), (RAM, df_makespan), (RAS, df_slowdown), (algo_random, df_random),
           (algo_first_fit, df_first_fit),
           (algo_tetris, df_tetris)]
    excludes = []
    # avg completions data using different training reward
    plot_single_stat_for_all_reward_signals(dfs, col_name=avg_completions, y_label="Average Completions",
                                            fig_dir=fig_dir, x_lable="Job Chunk Number")
    # avg makespans data using different training reward
    plot_single_stat_for_all_reward_signals(dfs, col_name=avg_makespans, y_label="Average Makespans", fig_dir=fig_dir,
                                            x_lable="Job Chunk Number")
    # avg slowdosns data using different training reward
    plot_single_stat_for_all_reward_signals(dfs, col_name=avg_slowdowns, y_label="Average Slowdowns", fig_dir=fig_dir,
                                            x_lable="Job Chunk Number")


def get_all_stats_df_old(range=0, exclude=None):
    df_completion = load_df(get_exp_file_path(file_type=avg_completions))
    df_makespan = load_df(get_exp_file_path(file_type=avg_makespans))
    df_slowdown = load_df(get_exp_file_path(file_type=avg_slowdowns))
    # df_others = load_df(get_exp_file_path(file_type=other_algo))
    df_tetris = load_df(get_exp_file_path(file_type=algo_tetris))
    df_random = load_df(get_exp_file_path(file_type=algo_random))
    df_first_fit = load_df(get_exp_file_path(file_type=algo_first_fit))
    if range > 0:
        return df_completion[:range], df_makespan[:range], df_slowdown[:range], \
               df_tetris[:range], df_random[:range], df_first_fit[:range]
    else:
        return df_completion, df_makespan, df_slowdown, df_tetris, df_random, df_first_fit


def get_all_stats_df(range=0, exclude=None):
    df_completion = load_df(get_exp_file_path(file_type=RAC))
    df_makespan = load_df(get_exp_file_path(file_type=RAM))
    df_slowdown = load_df(get_exp_file_path(file_type=RAS))
    df_mix_acas = load_df(get_exp_file_path(file_type=MIX_AC_AS))
    # df_others = load_df(get_exp_file_path(file_type=other_algo))
    df_tetris = load_df(get_exp_file_path(file_type=algo_tetris))
    df_random = load_df(get_exp_file_path(file_type=algo_random))
    df_first_fit = load_df(get_exp_file_path(file_type=algo_first_fit))
    if range > 0:
        return df_completion[:range], df_makespan[:range], df_slowdown[:range], df_mix_acas[:range], \
               df_tetris[:range], df_random[:range], df_first_fit[:range]
    else:
        return df_completion, df_makespan, df_slowdown, df_mix_acas, df_tetris, df_random, df_first_fit


# @exp_fig
def exp_stats_diff_by_reward_all_plots():
    """
    this plots the difference between two stats
    :return:
    """
    # if fig dir is specified, the figs will be save to the dir
    # fig_dir = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/figs/stats_diff"
    # fig_dir = None
    # df_completion = load_df(get_exp_file_path(file_type=avg_completions))
    # df_makespan = load_df(get_exp_file_path(file_type=avg_makespans))
    # df_slowdown = load_df(get_exp_file_path(file_type=avg_slowdowns))
    # # df_others = load_df(get_exp_file_path(file_type=other_algo))
    # df_tetris = load_df(get_exp_file_path(file_type=algo_tetris))
    # df_random = load_df(get_exp_file_path(file_type=algo_random))
    # df_first_fit = load_df(get_exp_file_path(file_type=algo_first_fit))
    df_completion, df_makespan, df_slowdown, df_tetris, df_random, df_first_fit = get_all_stats_df_old(range=180)

    # dfs = [(RAC, df_completion), (RAM, df_makespan), (RAS, df_slowdown), (algo_random, df_random),
    #        (algo_first_fit, df_first_fit),
    #        (algo_tetris, df_tetris)]
    dfs_makespans = [(RAC, df_completion), (RAS, df_slowdown), (algo_random, df_random),
                     (algo_first_fit, df_first_fit),
                     (algo_tetris, df_tetris)]
    dfs_completions = [(RAM, df_makespan), (RAS, df_slowdown), (algo_random, df_random),
                       (algo_first_fit, df_first_fit),
                       (algo_tetris, df_tetris)]
    dfs_slowdowns = [(RAC, df_completion), (RAM, df_makespan), (algo_random, df_random),
                     (algo_first_fit, df_first_fit),
                     (algo_tetris, df_tetris)]

    excludes = []
    suffix = "_range_180"
    # avg completions data using different training reward
    plot_stat_difference_for_all_reward_signals(dfs_completions, compare_stat=avg_completions,
                                                y_label="Average Completions Difference",
                                                fig_dir=fig_dir, df_compare_to=df_completion,
                                                compare_to_col_name=f"{reward_avg_completions}_{avg_completions}",
                                                suffix=suffix, x_lable="Job Chunk Number")
    # avg makespans data using different training reward
    plot_stat_difference_for_all_reward_signals(dfs_makespans, compare_stat=avg_makespans,
                                                y_label="Average Makespans Difference",
                                                fig_dir=fig_dir, df_compare_to=df_makespan,
                                                compare_to_col_name=f"{reward_avg_makespans}_{avg_makespans}",
                                                suffix=suffix, x_lable="Job Chunk Number")
    # # avg slowdosns data using different training reward
    plot_stat_difference_for_all_reward_signals(dfs_slowdowns, compare_stat=avg_slowdowns,
                                                y_label="Average Slowdowns Difference",
                                                fig_dir=fig_dir, df_compare_to=df_slowdown,
                                                compare_to_col_name=f"{reward_avg_slowdowns}_{avg_slowdowns}",
                                                suffix=suffix, x_lable="Job Chunk Number")


def get_job_stats(all_in_one_chart=False):
    df_jobs = load_df(job_file_path)
    save_dir = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/experiments/figs/job_stats"
    cols = ['duration', 'memory', 'instances_num']
    plot_all_job_stats_from_df(df_jobs, cols=cols, save_to=save_dir)  # this plots all stats in one chart
    # for col in cols:
    #     plot_individual_job_stats_from_df(df_jobs, cols=col, save_to=save_dir)


if __name__ == '__main__':
    # plot_avg_slowdown()
    # exp_results_by_reward_all_plots()
    # plot_reward_avg_completions()
    # plot_reward_avg_slowdowns()
    # rename_add_reward_type()
    # plot_training_stats_for_reward_all()
    # exp_results_by_reward_all_plots()
    # get_job_stats()
    hist_deepjs_avg()
