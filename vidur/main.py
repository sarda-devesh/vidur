from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()
    set_seeds(config.seed)

    # Verify that the replica and model counts are valid
    cluster_config = config.cluster_config
    replica_counts = len(cluster_config.num_replicas)
    model_counts = len(cluster_config.replica_config.model_names)
    if replica_counts != model_counts:
        raise Exception(f'Got counts for {replica_counts} models but only {model_counts} models specified')

    simulator = Simulator(config)
    simulator.run()

if __name__ == "__main__":
    main()
