from dataclasses import dataclass

import torch


@dataclass(slots=True)
class HiddenStateRecord:
    """
    Snapshot of hidden states captured for a single request/iteration.

    Attributes:
        request_id: Numeric identifier for the request.
        tokens: 1-D CPU tensor of token ids aligned with the hidden states.
        hidden_states: 2-D CPU tensor of hidden states (seq_len, hidden_size).
        next_token_id: The next token chosen by the target model (-100 if unknown).
        iteration: Decode iteration counter when the capture occurred.
    """

    request_id: int
    tokens: torch.Tensor
    hidden_states: torch.Tensor
    next_token_id: int
    iteration: int

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "tokens": self.tokens.tolist(),
            "hidden_states": self.hidden_states.tolist(),
            "next_token_id": self.next_token_id,
            "iteration": self.iteration,
        }
