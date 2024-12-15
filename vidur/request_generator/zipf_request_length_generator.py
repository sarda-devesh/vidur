from typing import Tuple

from vidur.config import ZipfRequestLengthGeneratorConfig
from vidur.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)
from vidur.utils.zipf_generator import ZipfGenerator


class ZipfRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(self, config: ZipfRequestLengthGeneratorConfig):
        super().__init__(config)

        self.zipf_generator = ZipfGenerator(
            config.min_tokens,
            config.max_tokens,
            config.theta,
            config.scramble,
            config.seed,
        )

    def get_next_num_tokens(self) -> Tuple[float, float]:
        prefill_tokens = self.zipf_generator.next()
        decode_tokens = self.zipf_generator.next()
        return prefill_tokens, decode_tokens
