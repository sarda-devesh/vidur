from utils import *
import shutil

def create_scan_configs(args):
    # Get the configs for batched
    all_scheduler_configs = []
    for max_bin_size in [1, 2, 4, 8, 16]:
        for binning_timeout in [0.25, 0.5, 1.0, 2.0, 4.0]:
            all_scheduler_configs.append({
                "scheduler_type" : "lor_batched",
                "max_bin_size" : max_bin_size,
                "binning_timeout" : binning_timeout
            })
    
    # Get the configs for combined
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            all_scheduler_configs.append({
                "scheduler_type" : "combined_balanced",
                "alpha" : alpha,
                "beta" : beta
            })

    # Create all config details
    num_requests = 1024
    workload_configs = [get_workload_config(config_type, num_requests) for config_type in ["zipfian", "synthetic"]]

    # Save all of the results
    write_configs_to_dir({
        "scheduler_config" : all_scheduler_configs,
        "workload_config" : workload_configs
    }, args.config_dir)

def run_experiment(args):
    num_workers = int(os.cpu_count()/5)
    run_all_configs_in_dir(args, num_workers)

def plot_results(args):
    print("Plot results called with args", args)

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