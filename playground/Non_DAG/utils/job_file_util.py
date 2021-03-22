import pandas as pd

job_attr = ["submit_time", "duration", "cpu", "memory", "job_id", "task_id", "instances_num", "disk"]


# batch_task columns
# create_timestamp: the create time of a task
# modify_timestamp: latest state modification time
# job_id
# task_id
# instance_num: number of instances for the task
# status: Task states includes Ready | Waiting | Running | Terminated | Failed | Cancelled
# plan_cpu: cpu requested for each instane of the task
# plan_mem: normalized memory requested for each instance of the task
#
def load_df(csv_path=None, ignore_idx=False):
    if csv_path:
        if ignore_idx:
            df = pd.read_csv(csv_path, index_col=[0])
        else:
            df = pd.read_csv(csv_path)
        return df


def collect_job_info():
    batch_task_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/data/trace_201708/batch_task_with_header.csv"
    save_to = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/jobs_files/jobs_2017.csv"
    df_task = load_df(csv_path=batch_task_path)
    print("Ths df_task len: ", len(df_task))

    df_job = pd.DataFrame()
    df_job['submit_time'] = df_task['create_ts']
    df_job['duration'] = df_task['modify_ts'] - df_task['create_ts']
    df_job['cpu'] = df_task['cpu'] / 100
    df_job['memory'] = df_task['memory']
    df_job['job_id'] = df_task['job_id']
    df_job['task_id'] = df_task['task_id']
    df_job['instances_num'] = df_task['instance_num']
    df_job['disk'] = [0] * len(df_task)
    print(f"get df_job len: {len(df_job)}")
    df_job.dropna(inplace=True)
    print(f"after dropna: {len(df_job)}")
    df_job = df_job[df_job['submit_time'] >= 0].copy()
    print(f"after drop negative ts: {len(df_job)}")
    df_job.to_csv(save_to)


def peek_data(job_path=""):
    df = load_df(job_path)
    print(f"len: {len(df)}")
    jobids = df['job_id'].nunique()
    print(f"unique job_id: {jobids}")


def check_data():
    path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/jobs_files/jobs_2017.csv"
    job_path = "/Users/jackz/Documents/P_Macbook/Laptop/Git_Workspace/DataScience/MachineLearning/MyForks/CloudSimPy/playground/Non_DAG/jobs_files/jobs.csv"
    peek_data(job_path)


if __name__ == '__main__':
    # collect_job_info()
    check_data()
