from typing import List, Tuple
import heapq
from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

class OutputReplicaWorkDone:
    def __init__(self, replica_id):
        self.replica_id = replica_id
        self.work_done = 0
    
    def __repr__(self):
        return f'Replica {self.replica_id} Work Done {self.work_done}'
    
    def __lt__(self, other):
        return self.work_done < other.work_done

class OutputGlobalScheduler(BaseGlobalScheduler):
    """
    Balance the requests by inputs
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workers = [OutputReplicaWorkDone(i) for i in range(self._num_replicas)]
        heapq.heapify(self.workers)
        print("Initialized the output global scheduler with", len(self.workers), "replicas")

    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            request_input_size, request_output_size = request.size
            
            # Assign the current job to the worker who has done the least amount of worker
            curr_worker = self.workers[0]
            request_mapping.append((curr_worker.replica_id, request))

            # Now update the work done for this replica
            curr_worker.work_done += request_output_size
            heapq.heapify(self.workers)

        return request_mapping