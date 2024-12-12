import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_summary():
    curr_dir = os.getcwd()
    colors = list(mcolors.TABLEAU_COLORS)
    color_idx = 0

    curr_colors = {}
    config_to_save = {}
    for dir_name in os.listdir(curr_dir):
        # Ensure this is a directory
        dir_path = os.path.join(curr_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
            
        for subdir_name in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir_name)
            if not os.path.isdir(subdir_path):
                continue
            
            # Load the current config
            config_path = os.path.join(subdir_path, "config.json")
            with open(config_path, 'r') as reader:
                curr_config = json.load(reader)

            # Get the current config details
            cluster_config = curr_config["cluster_config"]
            num_replicas = cluster_config["num_replicas"]
            scheduler_name = cluster_config["global_scheduler_config"]["name"]
            scheduler_name = " ".join(scheduler_name.split("_")).title()
            model_name = cluster_config["replica_config"]["model_name"]
            model_name = model_name.split("/")[1].strip()
            
            # Determine the linestyle and the color
            line_style = "-"
            if "8B" in model_name:
                line_style = "--"
            
            if scheduler_name not in curr_colors:
                curr_colors[scheduler_name] = colors[color_idx]
                color_idx += 1
            line_color = curr_colors[scheduler_name]
            
            # Save the result
            if num_replicas not in config_to_save:
                config_to_save[num_replicas] = {}
            replicas_details = config_to_save[num_replicas]

            key_name = scheduler_name + "," + model_name
            replicas_details[key_name] = (subdir_path, line_style, line_color)
    
    # Save the result
    with open("config_summary.json", "w+") as writer:
        json.dump(config_to_save, writer, indent=4, sort_keys=True)

if __name__ == "__main__":
    create_summary()