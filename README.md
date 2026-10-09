# quotient_filter

A small, dependency-free Python implementation of a quotient filter: a
probabilistic set-membership structure that stores hashed fingerprints and
answers membership queries with no false negatives and a bounded false
positive rate.

## Installation

Copy `src/quotient_filter/` onto your `PYTHONPATH` (or `pip install -e .` once
packaged). No third-party dependencies are required.

## Usage

```python
from quotient_filter import QuotientFilter

qf = QuotientFilter(capacity=1024, remainder_bits=8, seed=0)
for word in ["alpha", "beta", "gamma"]:
    qf.insert(word)

assert "alpha" in qf        # true (no false negatives)
assert qf.count == 3
assert qf.load_factor == 3 / 1024

# Probe an item that was never inserted; it may report present (false positive)
# but never reports a real member absent.
if "delta" in qf:
    print("possible false positive (or it was inserted earlier)")
```

Exported names:

- `QuotientFilter` — the only public class.

Constructor: `QuotientFilter(capacity=1024, remainder_bits=8, *, seed=None)`.
`capacity` is rounded up to the next power of two. `seed` fixes the hash salt
for deterministic behaviour; omit it for an independently-randomised filter.

Methods:

- `qf.insert(item) -> bool` — inserts; returns `True` on success, `False` if
  the table is full.
- `item in qf` — membership query.
- `qf.count` / `len(qf)` — number of inserted fingerprints.
- `qf.capacity` — number of buckets.
- `qf.load_factor` — `count / capacity`, a float in `[0.0, 1.0]`.
- `qf.false_positive_rate()` — analytic upper bound on the FP rate for an
  item not in the set, as a float in `[0.0, 1.0]`.
- `iter(qf)` — yields the stored remainder fingerprints (ints), not the
  original items; useful for serialization and tests.

## Why this exists

A quotient filter answers the same question as a Bloom filter ("might this item
be in the set?") but stores fingerprints in a way that keeps the load factor
visible and supports resize/merge operations cleanly. The trade-off here is
simplicity over throughput: rather than implementing the full canonical
run-walking algorithm with three status bits and cluster bookkeeping, this
implementation uses linear probing with a full-table scan on lookup. For
filters up to a few thousand buckets that is fast enough and far easier to
audit. The cost is that lookup is O(n) in the worst case rather than O(1)
amortised, and the filter does not support deletion or resize.

## The awkward edge

The filter does **not** deduplicate: inserting the same key twice consumes
two buckets. If you need set semantics, track membership yourself before
inserting. Also, because the lookup scan matches any stored remainder against
the queried remainder (without tracking which quotient each remainder belongs
to), the false-positive rate is the scan-based bound `1 - (1 - 1/R)^N`, not
the tighter quotient-aware bound a full canonical-run implementation achieves.
Plan capacity accordingly.
