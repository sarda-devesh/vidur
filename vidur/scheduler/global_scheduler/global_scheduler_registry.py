from vidur.scheduler.global_scheduler.lor_global_scheduler import LORGlobalScheduler
from vidur.scheduler.global_scheduler.lor_batched_global_scheduler import LORBatchedGlobalScheduler
from vidur.scheduler.global_scheduler.input_global_scheduler import InputGlobalScheduler
from vidur.scheduler.global_scheduler.output_global_scheduler import OutputGlobalScheduler
from vidur.scheduler.global_scheduler.combined_global_scheduler import CombinedGlobalScheduler
from vidur.scheduler.global_scheduler.random_global_scheduler import (
    RandomGlobalScheduler,
)
from vidur.scheduler.global_scheduler.round_robin_global_scheduler import (
    RoundRobinGlobalScheduler,
)
from vidur.types import GlobalSchedulerType
from vidur.utils.base_registry import BaseRegistry


class GlobalSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> GlobalSchedulerType:
        return GlobalSchedulerType.from_str(key_str)


GlobalSchedulerRegistry.register(GlobalSchedulerType.RANDOM, RandomGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.ROUND_ROBIN, RoundRobinGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR, LORGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.LOR_BATCHED, LORBatchedGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.INPUT_BALANCE, InputGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.OUTPUT_BALANCE, OutputGlobalScheduler)
GlobalSchedulerRegistry.register(GlobalSchedulerType.COMBINED_BALANCED, CombinedGlobalScheduler)