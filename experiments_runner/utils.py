import os
import subprocess
import json
import itertools
import argparse
import hashlib
from multiprocessing import Pool
import shutil

DEFAULT_CONFIGS = {
    "cluster_config" : {
        "device" : "a100"
    }, 
    "replica_config" : {
        "replica_names" : ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"],
        "replica_counts" : [1, 1]
    }, 
    "workload_config" : {
        "workload_type" : "synthetic"
    },
    "scheduler_config" : {
        "scheduler_type" : "random"
    }
}

def add_defaults_to_config(curr_config):
    for subconfig_key in DEFAULT_CONFIGS:
        subconfig_defaults = DEFAULT_CONFIGS[subconfig_key]
        if subconfig_key not in curr_config:
            curr_config[subconfig_key] = subconfig_defaults
            continue

        curr_subconfig_values = curr_config[subconfig_key]
        for subconfig_name in subconfig_defaults:
            if subconfig_name not in curr_subconfig_values:
                curr_subconfig_values[subconfig_name] = subconfig_defaults[subconfig_name]

def get_hash_of_config(config):
    config_str = json.dumps(config, sort_keys=True)
    hash_object = hashlib.sha256(config_str.encode())
    return  hash_object.hexdigest()

def write_configs_to_dir(config_combos, config_dir):
    keys, values = zip(*config_combos.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for curr_permutation in permutations_dicts:
        # Create the current config
        curr_combo = {}
        for config_type in curr_permutation:
            curr_combo[config_type] = curr_permutation[config_type]
        add_defaults_to_config(curr_combo)

        # Save the result
        curr_combo_hash = get_hash_of_config(curr_combo)
        save_path = os.path.join(config_dir, str(curr_combo_hash) + ".json")
        with open(save_path, "w+") as writer:
            json.dump(curr_combo, writer, indent=4, sort_keys=True)

def get_workload_config(config_type, num_requests = 1024):
    if config_type == "trace":
        return {
            "workload_type" : "trace",
            "data_path" : "/ssd/dsarda/vidur/data/processed_traces/splitwise_conv.csv"
        }
    elif config_type == "zipfian":
        return {
            "workload_type" : "zipfian",
            "zipf_theta" : 0.99,
            "num_requests" : num_requests
        }
    elif config_type == "synthetic":
        return {
            "workload_type" : "synthetic",
            "num_requests" : num_requests
        }

def read_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_dir", type=str, required = True, help="The directory containing the config")
    parser.add_argument("--results_dir", type=str, required = True, help="The directory containing the results")
    parser.add_argument("--mode", type=str, default = "create", help="The mode to run the experiment in")
    return parser.parse_args()

def record_cluster_config(cluster_config_data, command_params):
    command_params["replica_config_device"] = cluster_config_data["device"]

def record_replica_config(replica_config_data, command_params):
    command_params["replica_config_model_name"] = " ".join([str(item) for item in replica_config_data["replica_names"]])
    command_params["cluster_config_num_replicas"] = " ".join([str(item) for item in replica_config_data["replica_counts"]])

def record_scheduler_config(scheduler_config_data, command_params):
    scheduler_type = scheduler_config_data["scheduler_type"]
    command_params["global_scheduler_config"] = scheduler_type

    if scheduler_type == "lor_batched":
        command_params["lor_batched_global_scheduler_config_max_bin_size"] = scheduler_config_data["max_bin_size"]
        command_params["lor_batched_global_scheduler_config_binning_timeout"] = scheduler_config_data["binning_timeout"]
    elif scheduler_type == "combined_balanced":
        command_params["combined_global_scheduler_config_alpha"] = scheduler_config_data["alpha"]
        command_params["combined_global_scheduler_config_beta"] = scheduler_config_data["beta"]

def record_workload_type(workload_config_data, command_params):
    workload_type = workload_config_data["workload_type"]
    if workload_type == "trace":
        command_params["request_generator_config_type"] = "trace_replay"
        command_params["trace_request_generator_config_trace_file"] = workload_config_data["data_path"]
    else:
        command_params["request_generator_config_type"] = "synthetic"
        command_params["synthetic_request_generator_config_num_requests"] = workload_config_data["num_requests"]
        if workload_type == "zipfian":
            command_params["length_generator_config_type"] = "zipf"
            command_params["zipf_request_length_generator_config_theta"] = workload_config_data["zipf_theta"]

def get_command_for_config(config_file):
    with open(config_file, 'r') as reader:
        config_data = json.load(reader)

    # Get the command params
    command_params = {}
    record_cluster_config(config_data["cluster_config"], command_params)
    record_replica_config(config_data["replica_config"], command_params)
    record_scheduler_config(config_data["scheduler_config"], command_params)
    record_workload_type(config_data["workload_config"], command_params)
    return command_params

def run_for_config(config_file, args):
    current_dir = os.getcwd()
    os.chdir("/ssd/dsarda/vidur")
    try:
        # First get the command parameters
        command_params = get_command_for_config(config_file)

        # Add in the save path
        config_name = os.path.basename(config_file)
        config_name = config_name[ : config_name.index(".")]
        save_dir = os.path.join(args.results_dir, config_name)
        os.makedirs(save_dir, exist_ok = True)
        command_params["metrics_config_output_dir"] = save_dir

        # Save the json to the directory
        json_copy_path = os.path.join(save_dir, "request_config_summary.json")
        shutil.copy(config_file, json_copy_path)
        
        # Run the command
        command_params_txt = [
            "--" + str(key) + " " + str(value)
            for key, value in command_params.items()
        ]
        command_to_run = "python -m vidur.main " + " ".join(command_params_txt)
        print("Running command", command_to_run)
        result = subprocess.run(command_to_run, shell = True, capture_output = True, text = True)
    finally:
        os.chdir(current_dir)
    
    return True

def run_all_configs_in_dir(args, num_workers = 5):
    configs_to_runs = []
    for file_name in os.listdir(args.config_dir):
        if "json" not in file_name:
            continue
        
        config_path = os.path.join(args.config_dir, file_name)
        configs_to_runs.append((config_path, args))
    
    # Create the worker pool
    with Pool(num_workers) as worker_pool:
        worker_pool.starmap(run_for_config, configs_to_runs)

# Constants for graph generation
LOAD_BALANCING_COLOR_MAPPING = {
    "random" : "tab:blue",
    "round_robin" : "tab:orange",
    "lor" : "tab:green",
    "lor_batched" : "tab:red",
    "combined_balanced" : "tab:purple"
}
MODEl_NAME_LINE_MAPPING = {
    "meta-llama/Llama-2-7b-hf" : "-",
    "meta-llama/Meta-Llama-3-8B" : "--"
}
METRIC_NAME_MAPPING = {
    "request_e2e_time" : "Request End to End Time",
    "prefill_e2e_time" : "Time to First Token"
}