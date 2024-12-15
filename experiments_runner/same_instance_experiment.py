from utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

def create_scan_configs(args):
    # Create all config details
    num_requests = 8192
    all_workload_configs = [get_workload_config(config_type, num_requests) for config_type in ["trace", "zipfian"]]

    # Set the replica config
    all_replica_configs = [
        {
            "replica_names" : ["meta-llama/Llama-2-7b-hf"],
            "replica_counts" : [4]
        }, 
        {
            "replica_names" : ["meta-llama/Meta-Llama-3-8B"],
            "replica_counts" : [4]
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

def plot_subplot(args, axis, metric_name, target_workload_type, metric_range):
    metric_title = METRIC_NAME_MAPPING[metric_name]
    all_rows = []
    for dir_name in os.listdir(args.results_dir):
        dir_path = os.path.join(args.results_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        # Get the request summary
        request_summary_path = os.path.join(dir_path, "request_config_summary.json")
        with open(request_summary_path, 'r') as reader:
            curr_request_summary = json.load(reader)
        
        # Ensure the correct workload type
        workload_type = curr_request_summary["workload_config"]["workload_type"]
        if workload_type != target_workload_type:
            continue
        
        # Get the linestyle
        model_type = curr_request_summary["replica_config"]["replica_names"][0]
        line_type = MODEl_NAME_LINE_MAPPING[model_type]

        # Get the color
        scheduler_type = curr_request_summary["scheduler_config"]["scheduler_type"]
        line_color = LOAD_BALANCING_COLOR_MAPPING[scheduler_type]
        scheduler_label = scheduler_type.replace("_", " ").title()

        # Get the metrics
        metrics_path = os.path.join(dir_path, "request_metrics.csv")
        metrics_df = pd.read_csv(metrics_path)
        metrics_values = metrics_df[metric_name].values

        # Plot the line 
        line_label = model_type + " with " + scheduler_label + " balancer"
        axis.ecdf(metrics_values, color = line_color, linestyle = line_type, label = line_label)
    
    axis.set_xlim(metric_range)
    axis.set_ylim((0.9, 1.0))
    axis.legend(fontsize = 11)
    axis.set_xlabel(metric_title, fontsize = 14)
    axis.set_ylabel("Factor of requests", fontsize = 14)
    axis.set_title(f'CDF of {metric_title} with 4 instances for {target_workload_type.title()} workload', fontsize = 16)

def plot_results(args):
    fig, axes = plt.subplots(2, 2, figsize = (20, 10), sharey = True)
    plot_subplot(args, axes[0, 0], "request_e2e_time", "trace", (5, 20))
    plot_subplot(args, axes[0, 1], "prefill_e2e_time", "trace", (0.2, 1.2))
    plot_subplot(args, axes[1, 0], "request_e2e_time", "zipfian", (1.3, 3.0))
    plot_subplot(args, axes[1, 1], "prefill_e2e_time", "zipfian", (0.15, 0.4))

    # Save the result
    fig.suptitle('CDF of Key Metrics for Different Balancers', fontsize = 24)
    fig.tight_layout()
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