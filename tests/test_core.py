import math
import os
import sys

# Make `src` importable when running `python -m unittest discover -s tests`
# with PYTHONPATH=src (this fallback is for safety when that env var is unset).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest

from quotient_filter import QuotientFilter


class TestConstruction(unittest.TestCase):
    def test_default_construction(self):
        qf = QuotientFilter()
        self.assertEqual(qf.capacity, 1024)
        self.assertEqual(len(qf), 0)
        self.assertEqual(qf.count, 0)
        self.assertEqual(qf.load_factor, 0.0)

    def test_small_capacity_rounds_up_to_power_of_two(self):
        qf = QuotientFilter(capacity=3)
        # 3 rounds up to 4 (next power of two)
        self.assertEqual(qf.capacity, 4)

    def test_capacity_one(self):
        qf = QuotientFilter(capacity=1)
        self.assertEqual(qf.capacity, 1)
        self.assertTrue(qf.insert("a"))
        self.assertFalse(qf.insert("b"))

    def test_custom_remainder_bits(self):
        qf = QuotientFilter(capacity=4, remainder_bits=16)
        self.assertEqual(qf.capacity, 4)

    def test_invalid_capacity_raises(self):
        with self.assertRaises(ValueError):
            QuotientFilter(capacity=0)

    def test_invalid_remainder_bits_raises(self):
        with self.assertRaises(ValueError):
            QuotientFilter(remainder_bits=0)

    def test_seed_makes_filters_deterministic(self):
        a = QuotientFilter(seed=42)
        b = QuotientFilter(seed=42)
        # Same seed: same hash, so same fingerprints for the same item.
        self.assertEqual(a._hash("x"), b._hash("x"))

    def test_different_seeds_give_different_hashes(self):
        # Not a hard guarantee of hashing, but with 30 random bits and two
        # distinct seeds it holds overwhelmingly often. We use distinct,
        # widely-spaced seeds to make collision effectively impossible.
        a = QuotientFilter(seed=1)
        b = QuotientFilter(seed=100000)
        self.assertNotEqual(a._hash("hello"), b._hash("hello"))


class TestMembership(unittest.TestCase):
    def test_empty_filter_has_no_members(self):
        qf = QuotientFilter(capacity=16, seed=0)
        self.assertFalse("anything" in qf)
        self.assertFalse(qf.__contains__("anything"))

    def test_inserted_item_is_present(self):
        qf = QuotientFilter(capacity=16, seed=0)
        self.assertTrue(qf.insert("alpha"))
        self.assertIn("alpha", qf)

    def test_non_inserted_item_absent(self):
        qf = QuotientFilter(capacity=16, seed=0)
        qf.insert("alpha")
        self.assertNotIn("beta", qf)

    def test_no_false_negatives(self):
        # The defining correctness property: every inserted item must query
        # as present, regardless of collisions.
        qf = QuotientFilter(capacity=32, seed=7)
        items = [f"item-{i}" for i in range(25)]
        for it in items:
            self.assertTrue(qf.insert(it), f"insert failed for {it}")
        for it in items:
            self.assertIn(it, qf, f"false negative for {it}")

    def test_fill_to_capacity(self):
        qf = QuotientFilter(capacity=8, seed=0)
        ok = 0
        for i in range(8):
            if qf.insert(i):
                ok += 1
        self.assertEqual(ok, 8)
        self.assertEqual(qf.count, 8)
        self.assertEqual(qf.load_factor, 1.0)
        # The filter is now full; further inserts must fail.
        self.assertFalse(qf.insert("overflow"))

    def test_insert_returns_false_when_full(self):
        qf = QuotientFilter(capacity=2, seed=0)
        self.assertTrue(qf.insert("a"))
        self.assertTrue(qf.insert("b"))
        self.assertFalse(qf.insert("c"))
        self.assertEqual(qf.count, 2)


class TestFalsePositives(unittest.TestCase):
    def _count_fps(self, qf, inserted, candidates):
        fps = 0
        for c in candidates:
            if c in qf and c not in inserted:
                fps += 1
        return fps

    def test_false_positive_rate_zero_when_empty(self):
        qf = QuotientFilter(capacity=16, remainder_bits=4, seed=0)
        self.assertEqual(qf.false_positive_rate(), 0.0)

    def test_false_positive_rate_is_bounded_by_load(self):
        # With R=16 remainder values and N items, FP rate upper bound is
        # 1 - (1 - 1/16)^N. For small N this is approximately N/16.
        qf = QuotientFilter(capacity=16, remainder_bits=4, seed=0)
        for i in range(4):
            qf.insert(i)
        upper = qf.false_positive_rate()
        # Sanity: the analytic upper bound for 4 items in R=16 is
        # 1 - (15/16)^4 = 1 - 0.7724... = 0.2275...
        expected_upper = 1.0 - (15.0 / 16.0) ** 4
        self.assertAlmostEqual(upper, expected_upper, places=6)

    def test_false_positives_occur_at_high_load(self):
        # At very high load with small remainder space, false positives are
        # expected and observable. We don't assert an exact count (hashes
        # vary), only that the rate is within the documented bound.
        qf = QuotientFilter(capacity=4, remainder_bits=2, seed=12345)
        inserted = set()
        for i in range(4):
            qf.insert(i)
            inserted.add(i)
        # 1k probes with items unlikely to collide with inserted ints
        probes = [f"probe-{i}" for i in range(1000)]
        fps = self._count_fps(qf, inserted, probes)
        observed = fps / 1000.0
        upper = qf.false_positive_rate()
        # Observed should not exceed the upper bound materially.
        # Allow a generous epsilon for sampling noise on a 4-entry filter.
        self.assertLessEqual(observed, upper + 0.10)


class TestIteration(unittest.TestCase):
    def test_iter_yields_remainders(self):
        qf = QuotientFilter(capacity=8, seed=0)
        qf.insert("x")
        qf.insert("y")
        remainders = list(qf)
        self.assertEqual(len(remainders), 2)
        # Each entry is the stored remainder (an int), not the original item.
        for r in remainders:
            self.assertIsInstance(r, int)

    def test_iter_empty(self):
        qf = QuotientFilter(capacity=8, seed=0)
        self.assertEqual(list(qf), [])


class TestEdgeCases(unittest.TestCase):
    def test_unhashable_item_raises(self):
        # Lists are unhashable; the underlying hash() call must propagate the
        # TypeError rather than silently swallowing it.
        qf = QuotientFilter(capacity=8, seed=0)
        with self.assertRaises(TypeError):
            qf.insert([1, 2, 3])
        with self.assertRaises(TypeError):
            _ = [1, 2, 3] in qf

    def test_insert_same_item_twice_uses_two_slots(self):
        # This implementation does not deduplicate: inserting the same key
        # twice consumes two buckets. Documented behaviour.
        qf = QuotientFilter(capacity=8, seed=0)
        self.assertTrue(qf.insert("dup"))
        self.assertTrue(qf.insert("dup"))
        self.assertEqual(qf.count, 2)
        self.assertIn("dup", qf)

    def test_hash_value_is_stable_within_filter(self):
        qf = QuotientFilter(capacity=8, seed=99)
        q1, r1 = qf._hash("key")
        q2, r2 = qf._hash("key")
        self.assertEqual((q1, r1), (q2, r2))

    def test_load_factor_in_range(self):
        qf = QuotientFilter(capacity=8, seed=0)
        for i in range(5):
            qf.insert(i)
        lf = qf.load_factor
        self.assertGreaterEqual(lf, 0.0)
        self.assertLessEqual(lf, 1.0)
        self.assertAlmostEqual(lf, 5 / 8, places=6)


if __name__ == "__main__":
    unittest.main()
