from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LORBatchedGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        scheduler_config : LorBatchedGlobalSchedulerConfig = self._config.cluster_config.global_scheduler_config
        self.max_bin_size = scheduler_config.max_bin_size
        self.timeout = scheduler_config.binning_timeout
        
        # Set the additional fields
        self.need_to_choose_replica = True
        self.curr_replica_id = -1
        self.bin_start_time = None
        self.bin_size = 0

        self.pending_requests_map = {
            replica_scheduler.replica_id: 0
            for replica_scheduler in self._replica_schedulers.values()
        }

    def schedule(self) -> List[Tuple[int, Request]]:
        # Sort the requests
        self.sort_requests()
        curr_request = self._request_queue.pop(0)

        # See if we need to choose a replica
        request_mapping = []
        if self.need_to_choose_replica:
            self.curr_replica_id = min(self.pending_requests_map.items(), key=lambda x: x[1])[0]
            self.bin_size = 0
            self.bin_start_time = curr_request.arrived_at
            self.need_to_choose_replica = False

        # Map the request to the replica
        request_mapping.append((self.curr_replica_id, curr_request))
        self.pending_requests_map[self.curr_replica_id] += 1
        self.bin_size += 1
        bin_at_capacity = self.bin_size >= self.max_bin_size
        hit_timeout = (curr_request.arrived_at - self.bin_start_time) >= self.timeout
        self.need_to_choose_replica = bin_at_capacity or hit_timeout

        return request_mapping