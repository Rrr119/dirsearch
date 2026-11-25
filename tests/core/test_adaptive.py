from unittest import TestCase

from lib.connection.response import NativeResponse
from lib.core.adaptive import AdaptivePruner
from lib.core.dictionary import Dictionary
from lib.core.option_defaults import default_options
from lib.core.wordlist_config import WordlistConfig


def miss_response(body: bytes = b"shared soft 404 page") -> NativeResponse:
    return NativeResponse(
        "https://example.test/missing",
        404,
        [("content-type", "text/plain")],
        body,
    )


def fingerprint(response: NativeResponse) -> tuple:
    return (response.status, response.type, "", response.length, hash(response.content))


def make_pruner(sample_size: int = 10, prune_ratio: float = 0.9) -> AdaptivePruner:
    return AdaptivePruner(
        sample_size=sample_size,
        prune_ratio=prune_ratio,
        fingerprint_for=fingerprint,
    )


def make_dictionary(paths: list[str]) -> Dictionary:
    dictionary = Dictionary(WordlistConfig.from_options(default_options()))
    dictionary.__setstate__((list(paths), 0, [], 0))
    return dictionary


class TestAdaptivePrunerDecisions(TestCase):
    def test_threshold_rounds_up_to_meet_ratio(self):
        pruner = make_pruner(sample_size=10, prune_ratio=0.9)
        self.assertEqual(pruner.threshold, 9)

        pruner = make_pruner(sample_size=3, prune_ratio=0.5)
        self.assertEqual(pruner.threshold, 2)

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            AdaptivePruner(sample_size=0, prune_ratio=0.9, fingerprint_for=fingerprint)
        with self.assertRaises(ValueError):
            AdaptivePruner(sample_size=10, prune_ratio=0.0, fingerprint_for=fingerprint)
        with self.assertRaises(ValueError):
            AdaptivePruner(sample_size=10, prune_ratio=1.1, fingerprint_for=fingerprint)

    def test_dominant_fingerprint_prunes_only_when_sample_completes(self):
        pruner = make_pruner(sample_size=5, prune_ratio=0.8)

        for _ in range(4):
            self.assertFalse(pruner.observe(miss_response()))

        self.assertTrue(pruner.observe(miss_response()))
        # The decision latches: later observations cannot re-trigger pruning.
        self.assertFalse(pruner.observe(miss_response()))
        self.assertTrue(pruner.pruned)

    def test_mixed_fingerprints_do_not_prune(self):
        pruner = make_pruner(sample_size=10, prune_ratio=0.9)

        for index in range(9):
            self.assertFalse(pruner.observe(miss_response()))
        # One genuinely different response keeps the dominant class below 90%.
        self.assertFalse(pruner.observe(miss_response(b"unique real content")))
        self.assertFalse(pruner.pruned)

    def test_reset_clears_directory_scoped_statistics(self):
        pruner = make_pruner(sample_size=3, prune_ratio=1.0)
        for _ in range(3):
            pruner.observe(miss_response())
        self.assertTrue(pruner.pruned)

        pruner.reset()
        self.assertFalse(pruner.pruned)
        self.assertEqual(pruner.observed, 0)
        self.assertEqual(pruner.counts, {})

    def test_native_filtered_batch_uses_a_dedicated_fingerprint(self):
        pruner = make_pruner(sample_size=10, prune_ratio=0.9)
        # Eight engine-filtered misses plus two Python-visible shared misses
        # must not merge fingerprints across the engine boundary.
        self.assertFalse(pruner.observe_filtered_batch(8))
        self.assertFalse(pruner.observe(miss_response()))
        self.assertFalse(pruner.observe(miss_response()))
        self.assertFalse(pruner.pruned)

        pruner.reset()
        self.assertTrue(pruner.observe_filtered_batch(10))
        self.assertTrue(pruner.pruned)

    def test_batch_observation_never_exceeds_sample_size(self):
        pruner = make_pruner(sample_size=5, prune_ratio=1.0)
        self.assertTrue(pruner.observe_filtered_batch(50))
        self.assertEqual(pruner.observed, 5)


class TestDictionaryPruning(TestCase):
    def test_prune_remaining_drops_only_outstanding_entries(self):
        dictionary = make_dictionary(["a", "b", "c", "d", "e"])
        self.assertEqual(next(dictionary), "a")
        self.assertEqual(next(dictionary), "b")

        removed = dictionary.prune_remaining()

        self.assertEqual(removed, 3)
        self.assertEqual(len(dictionary), 2)
        with self.assertRaises(StopIteration):
            next(dictionary)

    def test_dynamic_extras_survive_pruning(self):
        dictionary = make_dictionary(["a", "b", "c"])
        dictionary.add_extra("discovered/")
        dictionary.prune_remaining()

        self.assertEqual(len(dictionary), 0)
        self.assertEqual(next(dictionary), "discovered/")

    def test_prune_is_idempotent_when_nothing_remains(self):
        dictionary = make_dictionary(["a"])
        self.assertEqual(dictionary.prune_remaining(), 1)
        self.assertEqual(dictionary.prune_remaining(), 0)

    def test_prune_with_active_claim_keeps_claims_recoverable(self):
        dictionary = make_dictionary(["a", "b", "c", "d"])
        claimed = dictionary.claim_next()
        self.assertEqual(claimed, "a")

        removed = dictionary.prune_remaining()

        self.assertEqual(removed, 3)
        # The in-flight claim remains owned until released.
        dictionary.release_claim(claimed)
        with self.assertRaises(StopIteration):
            dictionary.claim_next()
