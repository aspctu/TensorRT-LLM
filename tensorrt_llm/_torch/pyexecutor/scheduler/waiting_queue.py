from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from typing import Callable, Optional

from tensorrt_llm.llmapi.llm_args import WaitingQueuePolicy

from ..executor_request_queue import RequestQueueItem


class WaitingQueue(ABC):
    """Abstract base class for waiting queues."""

    @abstractmethod
    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> RequestQueueItem:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue."""
        pass

    @abstractmethod
    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSWaitingQueue(deque, WaitingQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: RequestQueueItem) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Add multiple requests to the queue according to FCFS policy."""
        self.extend(requests)

    def pop_request(self) -> RequestQueueItem:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> RequestQueueItem:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: RequestQueueItem) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        """Prepend all requests from another iterable to the front of this queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` iterable.
        """
        self.extendleft(requests)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        """Remove requests with the given IDs."""
        filtered_requests = [req for req in self if req.id not in request_ids]
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[RequestQueueItem]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityWaitingQueue(WaitingQueue):
    """A simple priority queue with stable FIFO tie-breaking."""

    def __init__(self, priority_fn: Callable[[RequestQueueItem], float]):
        self._items: list[RequestQueueItem] = []
        self._priority_fn = priority_fn
        self._next_arrival_order = 0

    def _track_request(self, request: RequestQueueItem) -> None:
        if not hasattr(request, "_priority_arrival_order"):
            request._priority_arrival_order = self._next_arrival_order
            self._next_arrival_order += 1

    def _score(self, request: RequestQueueItem) -> float:
        return self._priority_fn(request)

    def _best_index(self) -> int:
        if not self._items:
            raise IndexError("queue is empty")
        return max(
            range(len(self._items)),
            key=lambda idx: (
                self._score(self._items[idx]),
                -getattr(self._items[idx], "_priority_arrival_order", idx),
                -idx,
            ),
        )

    def add_request(self, request: RequestQueueItem) -> None:
        self._track_request(request)
        self._items.append(request)

    def add_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        for request in requests:
            self.add_request(request)

    def pop_request(self) -> RequestQueueItem:
        idx = self._best_index()
        return self._items.pop(idx)

    def peek_request(self) -> RequestQueueItem:
        return self._items[self._best_index()]

    def prepend_request(self, request: RequestQueueItem) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: Iterable[RequestQueueItem]) -> None:
        self.add_requests(requests)

    def remove_by_ids(self, request_ids: set[int]) -> None:
        self._items = [req for req in self._items if req.id not in request_ids]

    def __bool__(self) -> bool:
        return len(self._items) > 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[RequestQueueItem]:
        return iter(
            sorted(
                self._items,
                key=lambda request: (
                    self._score(request),
                    -getattr(request, "_priority_arrival_order", 0),
                    -request.id,
                ),
                reverse=True,
            )
        )


def create_waiting_queue(
    policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
    priority_fn: Optional[Callable[[RequestQueueItem], float]] = None,
) -> WaitingQueue:
    """Create a waiting queue based on the specified policy.

    Args:
        policy: The scheduling policy to use. Currently only FCFS is supported.
        priority_fn: Optional priority function for priority-aware queue ordering.

    Returns:
        A WaitingQueue instance.
    """
    if policy == WaitingQueuePolicy.FCFS:
        if priority_fn is not None:
            return PriorityWaitingQueue(priority_fn)
        return FCFSWaitingQueue()
    else:
        raise ValueError(f"Unsupported waiting queue policy: {policy}")
