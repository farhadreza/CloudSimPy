from abc import ABC


class RewardGiver(ABC):
    def __init__(self):
        self.simulation = None

    def attach(self, simulation):
        self.simulation = simulation

    def get_reward(self):
        if self.simulation is None:
            raise ValueError('Before calling method get_reward, the reward giver '
                             'must be attach to a simulation using method attach.')


class MakespanRewardGiver(RewardGiver):
    name = 'Makespan'

    def __init__(self, reward_per_timestamp):
        super().__init__()
        self.reward_per_timestamp = reward_per_timestamp

    def get_reward(self):
        super().get_reward()
        return self.reward_per_timestamp


class AverageSlowDownRewardGiver(RewardGiver):
    name = 'AS'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_tasks = cluster.unfinished_tasks
        reward = 0
        # ??????
        # this looks like if we want to maximize the reward we need to maximize the total duration, which doesn't make sense
        for task in unfinished_tasks:
            reward += (- 1 / task.task_config.duration)
        return reward


class MyAverageSlowDownRewardGiver(RewardGiver):
    name = 'MyAS'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_tasks = cluster.unfinished_tasks
        reward = 0
        for task in unfinished_tasks:
            reward += 1 / task.task_config.duration
        return reward


class AverageCompletionRewardGiver(RewardGiver):
    name = 'AC'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        return - unfinished_task_len


class MyAverageCompletionRewardGiver(RewardGiver):
    name = 'My_AC'

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_task_len = len(cluster.unfinished_tasks)
        reward = 0
        for task in cluster.unfinished_tasks:
            reward += 1 / (task.finished_timestamp - task.started_timestamp)
        return reward


class AverageMix_RAC_RAS(RewardGiver):
    name = "Mix_AC_AS"

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_tasks = cluster.unfinished_tasks
        reward = 0
        for task in unfinished_tasks:
            reward += (- 1 / task.task_config.duration)
        unfinished_task_len = len(cluster.unfinished_tasks)
        reward -= unfinished_task_len
        return reward


class MyAverageMix_RAC_RAS(RewardGiver):
    name = "My_Mix_AC_AS"

    def get_reward(self):
        super().get_reward()
        cluster = self.simulation.cluster
        unfinished_tasks = cluster.unfinished_tasks
        reward = 0
        for task in unfinished_tasks:
            reward += (1 / task.task_config.duration)
        unfinished_task_len = len(cluster.unfinished_tasks)
        reward -= unfinished_task_len
        return reward
