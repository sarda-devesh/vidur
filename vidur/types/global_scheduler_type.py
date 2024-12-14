from vidur.types.base_int_enum import BaseIntEnum


class GlobalSchedulerType(BaseIntEnum):
    RANDOM = 1
    ROUND_ROBIN = 2
    LOR = 3
    LOR_BATCHED = 4
    INPUT_BALANCE = 5
    OUTPUT_BALANCE = 6
    COMBINED_BALANCED = 7
