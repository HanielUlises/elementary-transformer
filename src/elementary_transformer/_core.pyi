from typing import Any

import numpy as np

def version() -> str: ...
def num_configurations(size: int, k: int) -> int: ...
def refine(
    size_left: int,
    relations_left: list[tuple[int, np.ndarray]],
    size_right: int,
    relations_right: list[tuple[int, np.ndarray]],
    k: int,
    q_max: int,
) -> dict[str, Any]: ...
