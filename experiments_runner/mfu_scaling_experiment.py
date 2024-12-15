from utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

def create_scan_configs(args):
    # Create all config details
    num_requests = 8192
    all_workload_configs = [get_workload_config(config_type, num_requests=num_requests) for config_type in ["trace", "zipfian"]]

    # Set the replica config
    all_replica_configs = [
        {
            "replica_names" : ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"],
            "replica_counts" : [1, 1]
        }, 
        {
            "replica_names" : ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"],
            "replica_counts" : [2, 2]
        },
        {
            "replica_names" : ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"],
            "replica_counts" : [4, 4]
        }
    ]

    # Set the scheduler configs
    all_scheduler_configs = [
        {
            "scheduler_type" : "random"
        }, 
        {
            "scheduler_type" : "round_robin"
        },
        {
            "scheduler_type" : "lor"
        },
        {
            "scheduler_type" : "lor_batched",
            "max_bin_size" : 8,
            "binning_timeout" : 2.0
        },
        {
            "scheduler_type" : "combined_balanced",
            "alpha" : 0.25,
            "beta" : 1.0
        }
    ]

    # Save all of the results
    write_configs_to_dir({
        "replica_config" : all_replica_configs,
        "scheduler_config" : all_scheduler_configs,
        "workload_config" : all_workload_configs
    }, args.config_dir)

def run_experiment(args):
    num_workers = int(os.cpu_count()/2)
    run_all_configs_in_dir(args, num_workers)

def get_average_mfu_values(plots_dir):
    total_mfu_value, curr_count = 0.0, 0
    for file_name in os.listdir(plots_dir):
        if "_mfu.json" not in file_name:
            continue
        
        # Read the json
        mfu_path = os.path.join(plots_dir, file_name)
        with open(mfu_path, 'r') as reader:
            mfu_data = json.load(reader)
        
        for metric_name in mfu_data:
            if "mean" in metric_name:
                total_mfu_value += mfu_data[metric_name]
                curr_count += 1
    
    return 0.0 if curr_count == 0 else (1.0 * total_mfu_value)/curr_count


def plot_results(args):
    plt.figure(figsize=(12, 5))
    values_to_plot = {}
    for dir_name in os.listdir(args.results_dir):
        dir_path = os.path.join(args.results_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        # Get the request summary
        request_summary_path = os.path.join(dir_path, "request_config_summary.json")
        with open(request_summary_path, 'r') as reader:
            curr_request_summary = json.load(reader)
        
        # Get the values for the bar chart
        replica_count = curr_request_summary["replica_config"]["replica_counts"][0]
        average_mfu_value = get_average_mfu_values(os.path.join(dir_path, "plots"))

        # Get the key name
        workload_type = curr_request_summary["workload_config"]["workload_type"]
        scheduler_type = curr_request_summary["scheduler_config"]["scheduler_type"]
        scheduler_label = scheduler_type.replace("_", " ").title()
        key_name = workload_type.title() + " workload with " + scheduler_label + " balancer"

        # Record the value
        if key_name not in values_to_plot:
            values_to_plot[key_name] = [
                LOAD_BALANCING_COLOR_MAPPING[scheduler_type], 
                WORKLOAD_SHADING_MAPPING[workload_type],
                [0.0, 0.0, 0.0]
            ]
        values_to_plot[key_name][2][replica_count // 2] = average_mfu_value

    # Graph the values
    x_labels = [1, 2, 4]
    x = 2 * np.arange(len(x_labels))
    width = 0.15
    multiplier = 0

    curr_attributes = list(values_to_plot.keys())
    curr_attributes.sort(key = lambda name : name.split(" ")[3])
    print(curr_attributes)
    for attribute in curr_attributes:
        attribute_values = values_to_plot[attribute]
        curr_color, shading, measurement = attribute_values[0], attribute_values[1], attribute_values[2]
        offset = width * multiplier
        plt.bar(x + offset, measurement, width, label = attribute, color = curr_color, hatch = shading)
        multiplier += 1
    
    # Set the labels
    label_offset = width * len(values_to_plot)/2.0
    plt.xticks(x + label_offset, x_labels)
    plt.xlabel("Number of Instances")
    plt.ylabel("Model MFU Utilization")
    plt.legend()
    plt.title("Average MFU Utilization with Instance Scaling")
    plt.tight_layout()
    
    # Save the result
    save_path = os.path.join(args.results_dir, "experiment_result.png")
    plt.savefig(save_path, dpi = 300)

def main():
    # Read in the args
    args = read_arguments()

    # Create the scan configs
    if args.mode == "create":
        if os.path.exists(args.config_dir):
            shutil.rmtree(args.config_dir)
        os.makedirs(args.config_dir, exist_ok = True)
        create_scan_configs(args)
    elif args.mode == "run":
        if os.path.exists(args.results_dir):
            shutil.rmtree(args.results_dir)
        os.makedirs(args.results_dir, exist_ok = True)
        run_experiment(args)
    elif args.mode == "plot":
        plot_results(args)

if __name__ == "__main__":
    main()