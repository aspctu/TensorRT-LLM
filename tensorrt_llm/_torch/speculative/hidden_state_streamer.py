from __future__ import annotations

import grpc
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch

logger = logging.getLogger(__name__)

TensorLike = Union[torch.Tensor, int, float]


@dataclass(frozen=True)
class HiddenStateDump:
    """Container for a single hidden-state dump."""

    request_id: Union[int, str]
    step: int
    hidden_states: torch.Tensor
    mtp_tokens: torch.Tensor
    accepted_tokens: torch.Tensor
    next_input_tokens: torch.Tensor
    next_token: int

    def as_serializable(self) -> Dict[str, object]:
        """Prepare a dictionary that can be serialized with torch.save()."""

        def _to_cpu_tensor(tensor: TensorLike) -> TensorLike:
            if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                return tensor.cpu()
            return tensor

        return {
            "request_id": str(self.request_id),
            "step": int(self.step),
            "hidden_states": _to_cpu_tensor(self.hidden_states),
            "mtp_tokens": _to_cpu_tensor(self.mtp_tokens),
            "accepted_tokens": _to_cpu_tensor(self.accepted_tokens),
            "next_input_tokens": _to_cpu_tensor(self.next_input_tokens),
            "next_token": int(self.next_token),
        }


class GrpcHiddenStateStreamer:
    """Streams hidden-state dumps to a remote gRPC endpoint."""

    def __init__(
        self,
        *,
        target: str,
        method: str,
        metadata: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[float] = None,
    ):
        if not method.startswith("/"):
            raise ValueError(
                f"gRPC method must be a fully-qualified path, got '{method}'."
            )

        self._channel = grpc.insecure_channel(target)

        self._rpc = self._channel.unary_unary(
            method,
            request_serializer=lambda payload: payload,
            response_deserializer=lambda data: data,
        )
        self._metadata = (
            tuple(metadata.items()) if metadata is not None else None
        )
        self._timeout = timeout_seconds
        self._has_logged_failure = False

    @staticmethod
    def _serialize(dump: HiddenStateDump) -> bytes:
        buffer = io.BytesIO()
        torch.save(dump.as_serializable(), buffer)
        return buffer.getvalue()

    def send(self, dump: HiddenStateDump) -> None:
        payload = self._serialize(dump)
        self._rpc(payload, metadata=self._metadata, timeout=self._timeout)

    def close(self) -> None:
        self._channel.close()

def build_hidden_state_streamer(
    config: Any,
) -> Optional[GrpcHiddenStateStreamer]:
    target = getattr(config, "hidden_state_stream_target", None)
    method = getattr(config, "hidden_state_stream_method", None)
    if not target or not method:
        return None

    metadata = getattr(config, "hidden_state_stream_metadata", None)
    timeout = getattr(config, "hidden_state_stream_timeout_seconds", None)

    logger.info(
        "Streaming MTP hidden-state dumps to gRPC endpoint %s%s.",
        target,
        method,
    )

    return GrpcHiddenStateStreamer(
        target=target,
        method=method,
        metadata=metadata,
        timeout_seconds=timeout,
    )

HiddenStateStreamer = GrpcHiddenStateStreamer
