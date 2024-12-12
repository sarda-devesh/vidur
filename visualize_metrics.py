import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns
import numpy as np

label_mapping = {
    "request_e2e_time" : ("Request End to End Time (sec)", "Request End to End Time"),
    "prefill_e2e_time" : ("Time to First Token (sec)", "Time to First Token")
}
load_balancer_mapping = {
    "Lor" : "Least Outstanding Requests",
    "Input Balance" : "Equally Expensive Requests"
}
def plot_metric(num_replicas, metric_name):
    config_path = "config_summary.json"
    with open(config_path, 'r') as reader:
        curr_config_summary = json.load(reader)
    
    plt.figure(figsize=(12, 6))
    replica_details = curr_config_summary[num_replicas]
    for key_name, key_details in replica_details.items():
        name_parts = key_name.split(",")
        lb_name = name_parts[0]
        if lb_name in load_balancer_mapping:
            lb_name = load_balancer_mapping[lb_name]
        legend_name = lb_name + " with " + name_parts[1]
        if "Output Balance" in name_parts[0]:
            continue

        dir_path, line_style, line_color = key_details[0], key_details[1], key_details[2]
        df = pd.read_csv(os.path.join(dir_path, 'request_metrics.csv'))
        
        # Graph the details
        sns.ecdfplot(data = df, x = metric_name, label = legend_name, color = line_color, linestyle = line_style)
    
    x_label, title_part = label_mapping[metric_name]
    plt.title(title_part + " with " + num_replicas + " replicas")
    plt.xlabel(x_label)
    plt.ylabel('Proportion of Requests')
    plt.legend()
    plt.tight_layout()

    save_name = os.path.join("visualizations", f'{metric_name}_{num_replicas}.png')
    plt.savefig(save_name, dpi = 300)

def plot_user_metrics():
    for num_replicas in ["2", "4", "8"]:
        for metric_name in ["request_e2e_time", "prefill_e2e_time"]:
            plot_metric(num_replicas, metric_name)

def get_mfu_for_dir(dir_path):
    plots_dir = os.path.join(dir_path, "plots")
    mfu_values = []
    for file_name in os.listdir(plots_dir):
        if "_mfu.json" not in file_name:
            continue
            
        file_path = os.path.join(plots_dir, file_name)
        with open(file_path, 'r') as reader:
            mfu_details = json.load(reader)
        
        for key_name, key_value in mfu_details.items():
            if "weighted_mean" in key_name:
                mfu_values.append(float(key_value))
    
    return sum(mfu_values)/len(mfu_values)

def plot_mfu_details():
    replica_process_order = ["2", "4", "8"]
    config_path = "config_summary.json"
    with open(config_path, 'r') as reader:
        curr_config_summary = json.load(reader)
    
    values_to_visualize = {}
    for num_replicas in replica_process_order:
        replica_details = curr_config_summary[num_replicas]
        for key_name, key_details in replica_details.items():
            name_parts = key_name.split(",")
            lb_name = name_parts[0]
            if lb_name in load_balancer_mapping:
                lb_name = load_balancer_mapping[lb_name]
            
            legend_name = lb_name + " with " + name_parts[1]
            if "Output Balance" in name_parts[0]:
                continue

            dir_path, line_style, line_color = key_details[0], key_details[1], key_details[2]
            dir_mean_mfu = get_mfu_for_dir(dir_path)

            if legend_name not in values_to_visualize:
                values_to_visualize[legend_name] = []
            values_to_visualize[legend_name].append(dir_mean_mfu)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(replica_process_order))
    width = 0.1
    curr_offset = 1

    multiplier = int((-1 * len(replica_process_order))/2)
    all_locs = []
    for attribute, measurement in values_to_visualize.items():
        offset = curr_offset + width * multiplier
        plot_locations = x + offset
        all_locs.append(plot_locations)
        rects = plt.bar(plot_locations, measurement, width, label=attribute)
        multiplier += 1
    
    plt.title("Average MFU Scaling for different load balancers", fontsize = 14)
    plt.xlabel("Number of Replicas")
    plt.ylabel("Average Model Flops Utilization")
    locs_idx = int(len(all_locs) // 2)
    loc_ticks = all_locs[locs_idx]
    plt.xticks(loc_ticks, replica_process_order)
    plt.legend()
    plt.tight_layout()

    save_name = os.path.join("visualizations", "mfu_mean_summary.png")
    plt.savefig(save_name, dpi = 300)

if __name__ == "__main__":
    plot_user_metrics()
    plot_mfu_details()