from utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

def create_scan_configs(args):
    # Get the configs for batched
    all_scheduler_configs = []
    for max_bin_size in [4, 8, 16, 32]:
        for binning_timeout in [1.0, 2.0, 4.0, 8.0]:
            all_scheduler_configs.append({
                "scheduler_type" : "lor_batched",
                "max_bin_size" : max_bin_size,
                "binning_timeout" : binning_timeout
            })
    
    # Get the configs for combined
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for beta in [0.25, 0.5, 0.75, 1.0]:
            all_scheduler_configs.append({
                "scheduler_type" : "combined_balanced",
                "alpha" : alpha,
                "beta" : beta
            })

    # Create all config details
    num_requests = 1024
    workload_configs = [get_workload_config(config_type, num_requests) for config_type in ["trace"]]

    # Set the replica config
    curr_replica_configs = [{
        "replica_names" : ["meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"],
        "replica_counts" : [2, 1]
    }]

    # Save all of the results
    write_configs_to_dir({
        "replica_config" : curr_replica_configs,
        "scheduler_config" : all_scheduler_configs,
        "workload_config" : workload_configs
    }, args.config_dir)

def run_experiment(args):
    num_workers = int(os.cpu_count()/2)
    run_all_configs_in_dir(args, num_workers)

SCHEDULE_TARGET_PARAMS = {
    "lor_batched" : ["binning_timeout", "max_bin_size"],
    "combined_balanced" : ["alpha", "beta"]
}


def plot_subplot(args, axis, target_schedule_type, metric_name):
    metric_title = METRIC_NAME_MAPPING[metric_name]
    schedule_params = SCHEDULE_TARGET_PARAMS[target_schedule_type]

    all_rows = []
    for dir_name in os.listdir(args.results_dir):
        dir_path = os.path.join(args.results_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        
        # Get the request summary
        request_summary_path = os.path.join(dir_path, "request_config_summary.json")
        with open(request_summary_path, 'r') as reader:
            curr_request_summary = json.load(reader)
        
        # Get the scheduler type
        scheduler_config = curr_request_summary["scheduler_config"]
        scheduler_type = scheduler_config["scheduler_type"]
        if scheduler_type != target_schedule_type:
            continue
        
        # Get the metrics
        metrics_path = os.path.join(dir_path, "request_metrics.csv")
        metrics_df = pd.read_csv(metrics_path)
        curr_metric_average = float(metrics_df[metric_name].quantile(0.99))

        # Create the current row
        curr_row_value = {"metric_val" : curr_metric_average}
        for param_name in schedule_params:
            param_col_name = param_name.replace("_", " ").title()
            curr_row_value[param_col_name] = scheduler_config[param_name]
        all_rows.append(curr_row_value)

    # Get the pivoted df
    curr_df = pd.DataFrame(all_rows)
    index_name = schedule_params[0].replace("_", " ").title()
    col_name = schedule_params[1].replace("_", " ").title()
    pivoted_df = pd.pivot_table(curr_df, values = 'metric_val', index = index_name, columns = col_name, aggfunc = "mean")
    summary_save_path = os.path.join(args.results_dir, f'{target_schedule_type}_{metric_name}.csv')
    pivoted_df.to_csv(summary_save_path)
    sns.heatmap(pivoted_df, ax = axis, annot=True)

    # Write the axis title
    method_name = target_schedule_type.replace("_", " ").title()
    curr_title = metric_title + " with " + method_name + " balancer"
    axis.set_title(curr_title)

def plot_results(args):
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    plot_subplot(args, axes[0], "lor_batched", "request_e2e_time")
    plot_subplot(args, axes[1], "combined_balanced", "request_e2e_time")

    # Save the result
    fig.suptitle('Parameter Search for Load Balancers', fontsize = 24)
    fig.tight_layout()
    save_path = os.path.join(args.results_dir, "experiment_result.png")
    plt.savefig(save_path, dpi = 300)

def main():
    # Read in the args
    args = read_arguments()

    # Create the scan configs
    if args.mode == "create":
        shutil.rmtree(args.config_dir)
        os.makedirs(args.config_dir, exist_ok = True)
        create_scan_configs(args)
    elif args.mode == "run":
        shutil.rmtree(args.results_dir)
        os.makedirs(args.results_dir, exist_ok = True)
        run_experiment(args)
    elif args.mode == "plot":
        plot_results(args)

if __name__ == "__main__":
    main()