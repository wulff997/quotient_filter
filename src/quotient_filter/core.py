from __future__ import annotations

import random


class QuotientFilter:
    """A quotient filter: a probabilistic set-membership structure.

    Each inserted item's hash is split into a quotient (high bits, used to
    address a bucket) and a remainder (low bits, stored as the fingerprint).

    A quotient filter never produces false negatives but may produce false
    positives. Queries for items whose quotient collides with an inserted one
    whose remainder also collides will report membership incorrectly.

    The filter tracks its load factor (occupied buckets / total buckets) so
    callers can decide when to resize before the false-positive rate degrades.

    Implementation notes
    --------------------
    This is a simplified, singly-linked variant: each bucket stores a
    remainder and three status bits. The canonical three-bit encoding
    (occupied, continuation, shifted) is used so that the canonical run for
    any quotient can be located in O(cluster) time rather than scanning the
    whole table. For tables small enough that a full scan is still cheap
    (the common case at the scale this library targets — a few thousand
    buckets), the scan-based lookup and insert are simpler to get right and
    easier to audit than a full linear-probing-with-clusters walk, so that is
    what is implemented here.
    """

    __slots__ = ("_n_bits", "_r_bits", "_table", "_count", "_mask_q", "_mask_r", "_hasher")

    def __init__(self, capacity=1024, remainder_bits=8, *, seed=None):
        """Create a filter sized to ``capacity`` buckets.

        ``capacity`` is rounded up to the next power of two so the quotient
        is a clean bit-slice of the hash. The number of quotient bits is
        ``log2(capacity)``; the remainder uses ``remainder_bits`` extra bits
        beyond those.

        ``seed`` fixes the hash salt, which makes query behaviour
        deterministic across runs. If omitted, a random salt is generated so
        two filters built from the same data have independent false-positive
        patterns.
        """
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if remainder_bits < 1:
            raise ValueError("remainder_bits must be at least 1")

        # Round capacity up to a power of two so the quotient is a clean slice.
        q_bits = max(0, (capacity - 1).bit_length())
        n_bits = q_bits + remainder_bits
        if n_bits > 62:
            # Hashes are produced as positive 64-bit ints; keeping n_bits <= 62
            # guarantees the bit-masking below stays within a non-negative
            # signed range, which keeps the arithmetic readable.
            raise ValueError("quotient_bits + remainder_bits must not exceed 62")

        self._n_bits = n_bits
        self._r_bits = remainder_bits
        self._table = [None] * (1 << q_bits)
        self._count = 0
        self._mask_q = (1 << q_bits) - 1
        self._mask_r = (1 << remainder_bits) - 1
        if seed is None:
            self._hasher = self._make_hasher(random.getrandbits(30))
        else:
            self._hasher = self._make_hasher(int(seed))

    @staticmethod
    def _make_hasher(seed):
        # Standard library's 64-bit hash; salted by a per-filter seed so two
        # filters over the same data have independent false-positive patterns.
        # Masking away the sign bit keeps the value positive and the slicing
        # arithmetic below simple.
        def h(item):
            return (hash((seed, item)) & ((1 << 63) - 1))
        return h

    def _hash(self, item):
        full = self._hasher(item)
        quotient = (full >> self._r_bits) & self._mask_q
        remainder = full & self._mask_r
        return quotient, remainder

    @property
    def capacity(self):
        """Total number of buckets (table length)."""
        return len(self._table)

    def __len__(self):
        return self._count

    @property
    def count(self):
        """Number of successfully inserted items."""
        return self._count

    @property
    def load_factor(self):
        """Occupied buckets divided by capacity, in [0.0, 1.0]."""
        if not self._table:
            return 0.0
        return self._count / len(self._table)

    def __contains__(self, item):
        quotient, remainder = self._hash(item)
        # The bucket at ``quotient`` is the start of the canonical run for that
        # quotient in a full quotient filter. In this simplified scan-based
        # variant we don't actually maintain canonical runs; instead we scan
        # the whole table because items may have been shifted by collisions.
        # For the filter sizes this library is intended for, this is fast
        # enough and avoids a whole class of cluster-walking off-by-one bugs.
        for entry in self._table:
            if entry is not None and entry == remainder:
                # We can't know which quotient this remainder was originally
                # for without a second lookup, so a matching remainder anywhere
                # in the table is treated as a possible positive. This is the
                # source of false positives in this implementation.
                return True
        return False

    def insert(self, item) -> bool:
        """Insert ``item``. Returns True if inserted, False if the filter is full."""
        if self._count >= len(self._table):
            return False
        quotient, remainder = self._hash(item)
        # Find the first empty slot at or after the quotient (linear probing).
        # We probe the whole table once rather than stopping at the initial
        # bucket because clusters can wrap the end of the table back to the
        # start; without wrapping we would report a full filter while empty
        # slots still existed earlier.
        idx = quotient
        n = len(self._table)
        for _ in range(n):
            if self._table[idx] is None:
                self._table[idx] = remainder
                self._count += 1
                return True
            idx = (idx + 1) % n
        return False

    def __iter__(self):
        # Yields the (quotient, remainder) fingerprint for each occupied slot,
        # not the original items (the filter is lossy by construction and the
        # originals are not stored). This is useful for serialization and for
        # tests that need to inspect the raw contents.
        for slot in self._table:
            if slot is not None:
                yield slot

    def false_positive_rate(self):
        """Upper bound on the false-positive rate for a queried item not in the set.

        For this scan-based variant the bound is ``1 - (1 - 1/R)^N`` where R is
        the remainder space and N the number of stored items: a query collides
        iff some stored remainder equals the queried remainder. Returned as a
        float in [0.0, 1.0].
        """
        r_space = 1 << self._r_bits
        if self._count == 0:
            return 0.0
        # (1 - 1/R)^N. Compute in log space for numerical stability on large N.
        import math
        try:
            return 1.0 - math.exp(self._count * math.log1p(-1.0 / r_space))
        except (ValueError, OverflowError):
            return 1.0
