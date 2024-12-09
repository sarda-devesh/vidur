#!/usr/bin/env python3
import json 
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import run, Popen, PIPE
import copy
from datetime import datetime
import glob

list_of_models = {
    "llama2-7b" : "meta-llama/Llama-2-7b-hf",
    "llama3-8b" : "meta-llama/Meta-Llama-3-8B",
    "llama3-70b" : "meta-llama/Meta-Llama-3-70B",
    "llama2-70b" : "meta-llama/Llama-2-70b-hf"
}

list_of_schedulers = {
    "round_robin" : "rr",
    "random" : "rand",
    "input_balance" : "ib",
    "output_balance" : "ob",
    "lor" : "lor"
}


def read_json_config(config, base_output_dir, scheduler_type):
    arguments = ["python3", "-m", "vidur.main", "--metrics_config_output_dir", f"{base_output_dir}"]
    models = []

    if (len(config['replicas']) != len(config['num_replicas'])):
        print("Invalid configuration!")
        print("List length of number of replicas must be the same as the list length of the type of replicas!")
        exit()

    for i, n in enumerate(config['replicas']):
        if (n in list_of_models):
            models.append(list_of_models[n])
            print(models)
        else:
            print("Invalid configuration!")
            print(f"Replica type of {i} is not of a valid type, must be {{llama2-7b, llama3-8b, llama3-70b, llama2-70b}}!")
            exit()
    
    arguments.append("--replica_config_device")
    arguments.append(config['device'])

    arguments.append("--global_scheduler_config")
    arguments.append(scheduler_type)

    arguments.append("--replica_config_model")
    arguments.append("--cluster_config_num_replicas")
    """ Going to use below to decouple both graphs for now... """
    arguments_2 = copy.deepcopy(arguments)

    arguments[4] = f"vidur_metrics_output_{config['replicas'][0]}_{list_of_schedulers[scheduler_type]}"
    arguments_2[4] = f"vidur_metrics_output_{config['replicas'][1]}_{list_of_schedulers[scheduler_type]}"
    print(arguments[4], arguments_2[4])

    rep_index = arguments.index("--replica_config_model") + 1
    arguments[rep_index:rep_index]  = [models[0]]

    nr_index = arguments.index("--cluster_config_num_replicas") + 1
    arguments[nr_index:nr_index]  = [config['num_replicas'][0]]

    arguments_2[rep_index:rep_index]  = [models[1]]
    arguments_2[nr_index:nr_index]  = [config['num_replicas'][1]]

    # e2e and prefill completion time
    print(arguments)
    print(arguments_2)
    
    return arguments, arguments_2

def metric_to_title(metric):
    return f'{metric.replace("_", " ").title()} with Two Instances'

def normalize(df, a, b):
    return a + ((df - df.min()) * (b - a)) / (df.max() - df.min()) 

def plot_cdfs(data_frames, metric, labels):
    max_val = -999
    min_val = 999
    index = 0
    for df in data_frames:
        if (df[metric].any() > max_val):
            max_val = df[metric].max()
        elif (df[metric].any() < min_val):
            min_val = df[metric].min()
    for df, label in zip(data_frames, labels):
        #df[metric] = normalize(df[metric], min_val, max_val)
        smoothed_data = pd.Series(df[metric]).rolling(window=5).mean()
        plt.plot(smoothed_data, df['cdf'], label=label)
    plt.legend(labels)
    plt.xlabel(f"{metric} (s)")
    plt.title(metric_to_title(metric))
    plt.ylabel('cdf')
    plt.show()



def plot_time_series(data_frames, metric, labels):
    print("ts")

def plot_avg_mfu(llms, labels):
    avg_mfus = []
    for llm in llms:
        mfu_data = glob.glob(f"{llm}/replica_*_stage_*_mfu.json")
        mfu = 0
        for file in mfu_data:
           json_id = file[len(llm) + 1:]
           with open(file, 'r') as data:
               mfu_config = json.load(data)
               mfu += mfu_config[f'{json_id.replace(".json", "")}_weighted_mean']
        avg_mfus.append(mfu / len(mfu_data))

    print(labels, avg_mfus)
    df = pd.DataFrame({'Average Model FLOPS' : avg_mfus}, index=labels)
    df.plot.bar(rot=0)
    plt.show()
    plt.cla()
    plt.clf()

def plot_metrics(config, base_output_dir, schedulers):
    llms = glob.glob(f"./{base_output_dir}_*/*/plots")
    labels = []
    data_frames = []

    """ Searching for all the output labels """
    for rep in config['replicas']:
        for sched in schedulers:
            benchmark = f"{rep}_{list_of_schedulers[sched]}"
            for llm in llms:
                index = -1
                if (llm.find(benchmark) != -1):
                    index = llm.find(benchmark)
                    labels.append(llm[index : index + len(benchmark)])

    """ For each of the highlighted metrics, let's graph those..."""
    for i,n in enumerate(config['metrics']):
        field_value = config['metrics'][n]
        if (int(field_value)):
            if (n == "avg_mfu"):
                plot_avg_mfu(llms, labels)
                continue
            for llm in llms:
                file = f'{llm}/{n}.csv'
                curr_df = pd.read_csv(file) 
                data_frames.append(curr_df)
            if ("time_series" in n):
                plot_time_series(data_frames, n, labels)
            else:
                plot_cdfs(data_frames, n, labels)
        del data_frames[:]
        

def main(filename, base_output_dir, schedulers, run=0):
    arguments = []
    processes = []
    with open(filename, 'r') as file:
        config = json.load(file)['config']
        for sched in schedulers:
            args, args_2 = read_json_config(config, base_output_dir, sched)
            arguments.append(args)
            arguments.append(args_2)

    if (run):
        for args in arguments:
            p = Popen(args, stdout=PIPE, stderr=PIPE, text=True)
            processes.append(p) 
    
        for p in processes:
            p.wait()
            print(p.stdout)
            print(p.stderr)

    plot_metrics(config, base_output_dir, schedulers)


if __name__ == "__main__":
    schedulers = ['lor', 'random', 'round_robin', 'input_balance', 'output_balance']
    main("config.json", "vidur_metrics_output", schedulers, run=0)
