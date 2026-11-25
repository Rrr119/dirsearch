"""Response-driven adaptive dictionary pruning.

The pruner observes the first ``sample_size`` responses of one dictionary
pass (one scanned directory). When a single response fingerprint dominates
that sample beyond ``prune_ratio``, the remaining dictionary entries are
considered wasteful: the target clearly answers with one shared "not found"
class for this directory, so the outstanding entries are skipped.

The component keeps only policy and counters; the :class:`~lib.core.dictionary.Dictionary`
owns the actual queue mutation. It is intentionally engine-agnostic so the
threaded, async and native fuzzers share the same decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from lib.connection.response import BaseResponse


ResponseFingerprint = Callable[[BaseResponse], tuple]


@dataclass(slots=True)
class AdaptivePruner:
    """Track response fingerprints during one dictionary pass.

    A fresh instance (or :meth:`reset`) is required for every scanned
    directory because each directory has its own wildcard/soft-404 behavior.
    ``pruned`` latches so a single pass can trigger the dictionary cut once.
    """

    sample_size: int
    prune_ratio: float
    fingerprint_for: ResponseFingerprint
    observed: int = 0
    counts: dict[tuple, int] = field(default_factory=dict)
    pruned: bool = False

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("adaptive sample size must be greater than zero")
        if not 0.0 < self.prune_ratio <= 1.0:
            raise ValueError("adaptive prune ratio must be between 0 and 1")

    def reset(self) -> None:
        """Forget the previous directory's fingerprint statistics."""
        self.observed = 0
        self.counts.clear()
        self.pruned = False

    @property
    def threshold(self) -> int:
        """Minimum number of identical fingerprints required to prune."""
        # Round up so the configured ratio is met or exceeded (e.g. ratio 0.9
        # with sample_size 10 needs 9 identical responses, not 8).
        return max(1, -(-self.sample_size * self.prune_ratio // 1))

    def observe(self, response: BaseResponse) -> bool:
        """Record one response and report whether the dictionary must be cut.

        Returns ``True`` at most once per pass. The decision only uses the
        first ``sample_size`` responses; later observations are ignored.
        """
        if self.pruned or self.observed >= self.sample_size:
            return False

        return self._record(self.fingerprint_for(response), 1)

    def observe_filtered_batch(self, count: int) -> bool:
        """Record a run of engine-side filtered misses under one fingerprint.

        The native request engine classifies soft-404 responses inside Rust and
        never materializes them in Python. They share one filter class, so the
        whole run is attributed to a dedicated fingerprint instead of building
        response objects that do not exist on this side of the boundary.
        """
        if count <= 0:
            return False
        return self._record(("native-filtered",), count)

    def _record(self, fingerprint: tuple, count: int) -> bool:
        if self.pruned or self.observed >= self.sample_size:
            return False

        count = min(count, self.sample_size - self.observed)
        self.observed += count
        current = self.counts.get(fingerprint, 0) + count
        self.counts[fingerprint] = current

        if self.observed >= self.sample_size and current >= self.threshold:
            self.pruned = True
            return True

        return False
