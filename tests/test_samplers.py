import numpy as np
import pytest
import torch

from narw_classifier.utils.samplers import make_balanced_sampler


class TestMakeBalancedSampler:
    def test_balances_imbalanced_classes(self):
        # 1 positive in 100 → without balancing the sampler should pick it ~1% of the
        # time; with balancing it should pick it ~50% of the time.
        labels = [1] + [0] * 99
        sampler = make_balanced_sampler(labels, seed=0)
        # Draw 10_000 samples and check the positive's selection rate.
        draws = list(iter(sampler))
        n_pos = sum(1 for i in draws if labels[i] == 1)
        rate = n_pos / len(draws)
        # Allow generous slack for sampling noise; rate should clearly be > 0.4.
        assert 0.4 < rate < 0.6, f"positive sample rate {rate:.2f} is not balanced"

    def test_deterministic_with_same_seed(self):
        labels = [0, 0, 0, 1, 1] * 10
        a = list(iter(make_balanced_sampler(labels, seed=42)))
        b = list(iter(make_balanced_sampler(labels, seed=42)))
        assert a == b

    def test_different_seeds_give_different_orders(self):
        labels = [0, 0, 0, 1, 1] * 10
        a = list(iter(make_balanced_sampler(labels, seed=1)))
        b = list(iter(make_balanced_sampler(labels, seed=2)))
        assert a != b

    def test_raises_when_a_class_is_missing(self):
        with pytest.raises(ValueError, match="both classes"):
            make_balanced_sampler([0, 0, 0, 0], seed=0)

    def test_accepts_numpy_array_labels(self):
        labels = np.array([0, 0, 1, 1, 0, 0, 1])
        sampler = make_balanced_sampler(labels, seed=0)
        assert isinstance(sampler.weights, torch.Tensor)
