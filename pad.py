import numpy as np
from ops import Ops, OPS_N


class Pad():
    def __init__(self) -> None:
        self.ops_n = OPS_N
        self.mapping = {}
        self.origin_space = -1

    def define_mapping(self, mapping: dict[int, str]) -> None:
        for key, value in mapping.items():
            try:
                self.mapping[Ops.__members__[value]] = key
            except KeyError:
                raise ValueError(f"Invalid operation {value}")
        self.origin_space = len(mapping)

    def mapped_ops(self, ops) -> np.ndarray:
        actions = np.zeros(self.origin_space)
        for id, value in enumerate(ops):
            if id in self.mapping:
                actions[self.mapping[id]] = value
        return actions
