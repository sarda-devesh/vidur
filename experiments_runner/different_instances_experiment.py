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
            "replica_counts" : [3, 1]
        }, 
        {
            "replica_names" : ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"],
            "replica_counts" : [2, 2]
        },
        {
            "replica_names" : ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"],
            "replica_counts" : [1, 3]
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

def plot_subplot(args, axis, target_replica_counts, metric_name, metric_range):
    metric_title = METRIC_NAME_MAPPING[metric_name]
    curr_axis_title = f'CDF of {metric_title} with ({target_replica_counts[0]}x Llama-2-7B, {target_replica_counts[1]}x Llama-3-8B)'

    for dir_name in os.listdir(args.results_dir):
        dir_path = os.path.join(args.results_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        # Get the request summary
        request_summary_path = os.path.join(dir_path, "request_config_summary.json")
        with open(request_summary_path, 'r') as reader:
            curr_request_summary = json.load(reader)
        
        # Ensure the correct workload type
        replica_counts = curr_request_summary["replica_config"]["replica_counts"]
        if replica_counts != target_replica_counts:
            continue
        
        # Get the linestyle
        workload_type = curr_request_summary["workload_config"]["workload_type"]
        line_type = WORKLOAD_LINE_MAPPING[workload_type]

        # Get the color
        scheduler_type = curr_request_summary["scheduler_config"]["scheduler_type"]
        line_color = LOAD_BALANCING_COLOR_MAPPING[scheduler_type]
        scheduler_label = scheduler_type.replace("_", " ").title()

        # Get the metrics
        metrics_path = os.path.join(dir_path, "request_metrics.csv")
        metrics_df = pd.read_csv(metrics_path)
        metrics_values = metrics_df[metric_name].values

        # Plot the line 
        line_label = workload_type.title() + " workload with " + scheduler_label + " balancer"
        axis.ecdf(metrics_values, color = line_color, linestyle = line_type, label = line_label)

    axis.set_xlim(metric_range)
    axis.set_ylim((0.9, 1.0))
    axis.xaxis.set_tick_params(labelsize=13)
    axis.yaxis.set_tick_params(labelsize=13)
    axis.set_xlabel(metric_title, fontsize = 16)
    axis.set_ylabel("Factor of requests", fontsize = 16)
    axis.set_title(curr_axis_title, fontsize = 18)

def plot_results(args):
    fig, axes = plt.subplots(1, 3, figsize = (30, 5), sharey = True)
    plot_subplot(args, axes[0], [1, 3], "request_e2e_time", (0, 35))
    plot_subplot(args, axes[1], [2, 2], "request_e2e_time", (0, 35))
    plot_subplot(args, axes[2], [3, 1], "request_e2e_time", (0, 35))
    handles,labels = axes[2].get_legend_handles_labels()

    # Save the result
    fig.suptitle('CDF of Request Time for Different Model Configurations', fontsize = 24)
    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.25)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(handles) // 2, fontsize = 13)
    save_path = os.path.join(args.results_dir, "experiment_result.png")
    print(save_path)
    plt.savefig(save_path, dpi = 300, bbox_inches='tight')

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
