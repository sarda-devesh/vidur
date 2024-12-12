import json

from vidur.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger

logger = init_logger(__name__)


class Cluster(BaseEntity):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Cluster.generate_id()
        self._config = cluster_config

        # get metrics config
        self._output_dir = metrics_config.output_dir

        # Init replica object handles
        self._replicas = {}

        replica_counts = self._config.num_replicas
        model_configs = self._config.replica_config.model_configs
        for idx in range(len(replica_counts)):
            curr_model_config = model_configs[idx]
            model_counts = replica_counts[idx]
            print("Creating", model_counts, "replicas of", self._config.replica_config.model_names[idx])
            for _ in range(model_counts):
                replica = Replica(self._config.replica_config, curr_model_config, generator_config)
                self._replicas[replica.id] = replica
        print("Got replica ids of", self._replicas.keys())

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()

    @property
    def replicas(self):
        return self._replicas

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
        }

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)
