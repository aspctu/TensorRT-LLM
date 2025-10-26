from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

HiddenStateObserver = Callable[['HiddenStateRecord'], None]


@dataclass
class HiddenStateRecord:
    request_id: int
    phase: str
    tokens: torch.Tensor
    hidden_states: torch.Tensor
    num_accepted: int
    iteration: int
    organization_id: Optional[Union[int, str]] = None
