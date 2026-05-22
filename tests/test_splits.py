import pytest

from narw_classifier.data.splits import stratified_split


def _make_data(n_pos: int, n_neg: int) -> tuple[list[str], list[int]]:
    files = [f"pos_{i}.aif" for i in range(n_pos)] + [f"neg_{i}.aif" for i in range(n_neg)]
    labels = [1] * n_pos + [0] * n_neg
    return files, labels


class TestStratifiedSplit:
    def test_class_proportions_preserved(self):
        files, labels = _make_data(n_pos=100, n_neg=900)
        tr_f, tr_y, va_f, va_y = stratified_split(files, labels, val_fraction=0.2, seed=0)
        # Roughly 10% positives in each split (stratified)
        assert abs(sum(tr_y) / len(tr_y) - 0.10) < 0.01
        assert abs(sum(va_y) / len(va_y) - 0.10) < 0.01

    def test_deterministic_with_same_seed(self):
        files, labels = _make_data(n_pos=50, n_neg=450)
        a = stratified_split(files, labels, val_fraction=0.2, seed=42)
        b = stratified_split(files, labels, val_fraction=0.2, seed=42)
        assert a == b

    def test_different_seeds_give_different_splits(self):
        files, labels = _make_data(n_pos=50, n_neg=450)
        a = stratified_split(files, labels, val_fraction=0.2, seed=1)
        b = stratified_split(files, labels, val_fraction=0.2, seed=2)
        assert a != b

    def test_no_overlap_between_train_and_val(self):
        files, labels = _make_data(n_pos=50, n_neg=450)
        tr_f, _, va_f, _ = stratified_split(files, labels, val_fraction=0.2, seed=0)
        assert set(tr_f).isdisjoint(set(va_f))
        assert set(tr_f) | set(va_f) == set(files)

    def test_invalid_val_fraction_raises(self):
        files, labels = _make_data(10, 10)
        with pytest.raises(ValueError):
            stratified_split(files, labels, val_fraction=0.0, seed=0)
        with pytest.raises(ValueError):
            stratified_split(files, labels, val_fraction=1.0, seed=0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            stratified_split(["a.aif", "b.aif"], [0], val_fraction=0.2, seed=0)
