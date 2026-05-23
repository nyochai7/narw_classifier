import pytest

from narw_classifier.data.splits import (
    day_stratified_split,
    day_stratified_split_3way,
    stratified_split,
    stratified_split_3way,
)


def _make_data(n_pos: int, n_neg: int) -> tuple[list[str], list[int]]:
    files = [f"pos_{i}.aif" for i in range(n_pos)] + [f"neg_{i}.aif" for i in range(n_neg)]
    labels = [1] * n_pos + [0] * n_neg
    return files, labels


def _make_dated_data(
    dates_with_counts: dict[str, tuple[int, int]],
) -> tuple[list[str], list[int]]:
    """Build filenames following the real NARW pattern.

    ``dates_with_counts``: ``{YYYYMMDD: (n_pos, n_neg)}``.
    """
    files: list[str] = []
    labels: list[int] = []
    idx = 0
    for date, (n_pos, n_neg) in dates_with_counts.items():
        for _ in range(n_pos):
            files.append(f"{date}_000000_000s0ms_TRAIN{idx}_1.aif")
            labels.append(1)
            idx += 1
        for _ in range(n_neg):
            files.append(f"{date}_000000_000s0ms_TRAIN{idx}_0.aif")
            labels.append(0)
            idx += 1
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


class TestDayStratifiedSplit:
    @staticmethod
    def _dates_in(files: list[str]) -> set[str]:
        return {f[:8] for f in files}

    def test_no_date_overlap_between_train_and_val(self):
        # 10 dates, each with 5 pos + 50 neg = 550 total clips
        dates = {f"2009040{d}": (5, 50) for d in range(10)}
        files, labels = _make_dated_data(dates)
        tr_f, _, va_f, _ = day_stratified_split(files, labels, val_fraction=0.3, seed=0)
        assert self._dates_in(tr_f).isdisjoint(self._dates_in(va_f))

    def test_no_clip_overlap_and_full_coverage(self):
        dates = {f"2009040{d}": (5, 50) for d in range(10)}
        files, labels = _make_dated_data(dates)
        tr_f, _, va_f, _ = day_stratified_split(files, labels, val_fraction=0.3, seed=0)
        assert set(tr_f).isdisjoint(set(va_f))
        assert set(tr_f) | set(va_f) == set(files)

    def test_both_classes_present_in_both_splits(self):
        dates = {f"2009040{d}": (5, 50) for d in range(10)}
        files, labels = _make_dated_data(dates)
        _, tr_y, _, va_y = day_stratified_split(files, labels, val_fraction=0.3, seed=0)
        assert set(tr_y) == {0, 1}
        assert set(va_y) == {0, 1}

    def test_deterministic_with_same_seed(self):
        dates = {f"2009040{d}": (5, 50) for d in range(10)}
        files, labels = _make_dated_data(dates)
        a = day_stratified_split(files, labels, val_fraction=0.3, seed=42)
        b = day_stratified_split(files, labels, val_fraction=0.3, seed=42)
        assert a == b

    def test_different_seeds_give_different_date_partitions(self):
        # 20 dates so there's enough room for distinct shuffles
        dates = {f"200904{d:02d}": (3, 20) for d in range(1, 21)}
        files, labels = _make_dated_data(dates)
        _, _, va_a, _ = day_stratified_split(files, labels, val_fraction=0.3, seed=1)
        _, _, va_b, _ = day_stratified_split(files, labels, val_fraction=0.3, seed=2)
        assert self._dates_in(va_a) != self._dates_in(va_b)

    def test_invalid_val_fraction_raises(self):
        files, labels = _make_dated_data({"20090401": (5, 5), "20090402": (5, 5)})
        with pytest.raises(ValueError):
            day_stratified_split(files, labels, val_fraction=0.0, seed=0)
        with pytest.raises(ValueError):
            day_stratified_split(files, labels, val_fraction=1.0, seed=0)

    def test_unparseable_filename_raises(self):
        with pytest.raises(ValueError, match="parse date"):
            day_stratified_split(["bogus.aif"], [1], val_fraction=0.2, seed=0)

    def test_raises_when_a_class_missing_from_a_split(self):
        # Construct an adversarial case: all positives on one date, all negatives elsewhere.
        # Any day-level split will starve one side of a class.
        files, labels = _make_dated_data(
            {"20090401": (20, 0), "20090402": (0, 50), "20090403": (0, 50)}
        )
        with pytest.raises(RuntimeError, match="missing a class"):
            day_stratified_split(files, labels, val_fraction=0.3, seed=0)


class TestStratifiedSplit3Way:
    def test_full_coverage_no_overlap(self):
        files, labels = _make_data(n_pos=100, n_neg=900)
        tr_f, _, va_f, _, te_f, _ = stratified_split_3way(
            files, labels, val_fraction=0.15, test_fraction=0.15, seed=0
        )
        assert set(tr_f) | set(va_f) | set(te_f) == set(files)
        assert set(tr_f).isdisjoint(set(va_f))
        assert set(tr_f).isdisjoint(set(te_f))
        assert set(va_f).isdisjoint(set(te_f))

    def test_approximate_70_15_15_fractions(self):
        files, labels = _make_data(n_pos=1000, n_neg=9000)
        tr_f, _, va_f, _, te_f, _ = stratified_split_3way(
            files, labels, val_fraction=0.15, test_fraction=0.15, seed=0
        )
        n = len(files)
        assert 0.68 < len(tr_f) / n < 0.72
        assert 0.13 < len(va_f) / n < 0.17
        assert 0.13 < len(te_f) / n < 0.17

    def test_class_proportions_preserved(self):
        files, labels = _make_data(n_pos=100, n_neg=900)
        _, tr_y, _, va_y, _, te_y = stratified_split_3way(
            files, labels, val_fraction=0.15, test_fraction=0.15, seed=0
        )
        for ys in (tr_y, va_y, te_y):
            assert abs(sum(ys) / len(ys) - 0.10) < 0.02

    def test_deterministic_with_same_seed(self):
        files, labels = _make_data(n_pos=50, n_neg=450)
        a = stratified_split_3way(files, labels, val_fraction=0.15, test_fraction=0.15, seed=42)
        b = stratified_split_3way(files, labels, val_fraction=0.15, test_fraction=0.15, seed=42)
        assert a == b

    def test_invalid_fractions_raise(self):
        files, labels = _make_data(10, 90)
        with pytest.raises(ValueError):
            stratified_split_3way(files, labels, val_fraction=0.0, test_fraction=0.15, seed=0)
        with pytest.raises(ValueError):
            stratified_split_3way(files, labels, val_fraction=0.15, test_fraction=0.0, seed=0)
        with pytest.raises(ValueError):
            stratified_split_3way(files, labels, val_fraction=0.5, test_fraction=0.5, seed=0)


class TestDayStratifiedSplit3Way:
    @staticmethod
    def _dates_in(files: list[str]) -> set[str]:
        return {f[:8] for f in files}

    def test_no_date_overlap_between_any_two_splits(self):
        dates = {f"200904{d:02d}": (3, 30) for d in range(1, 21)}
        files, labels = _make_dated_data(dates)
        tr_f, _, va_f, _, te_f, _ = day_stratified_split_3way(
            files, labels, val_fraction=0.15, test_fraction=0.15, seed=0
        )
        tr_d, va_d, te_d = self._dates_in(tr_f), self._dates_in(va_f), self._dates_in(te_f)
        assert tr_d.isdisjoint(va_d)
        assert tr_d.isdisjoint(te_d)
        assert va_d.isdisjoint(te_d)

    def test_full_coverage(self):
        dates = {f"200904{d:02d}": (3, 30) for d in range(1, 21)}
        files, labels = _make_dated_data(dates)
        tr_f, _, va_f, _, te_f, _ = day_stratified_split_3way(
            files, labels, val_fraction=0.15, test_fraction=0.15, seed=0
        )
        assert set(tr_f) | set(va_f) | set(te_f) == set(files)

    def test_both_classes_present_in_all_splits(self):
        dates = {f"200904{d:02d}": (3, 30) for d in range(1, 21)}
        files, labels = _make_dated_data(dates)
        _, tr_y, _, va_y, _, te_y = day_stratified_split_3way(
            files, labels, val_fraction=0.15, test_fraction=0.15, seed=0
        )
        for ys in (tr_y, va_y, te_y):
            assert set(ys) == {0, 1}

    def test_deterministic_with_same_seed(self):
        dates = {f"200904{d:02d}": (3, 30) for d in range(1, 21)}
        files, labels = _make_dated_data(dates)
        a = day_stratified_split_3way(files, labels, val_fraction=0.15, test_fraction=0.15, seed=42)
        b = day_stratified_split_3way(files, labels, val_fraction=0.15, test_fraction=0.15, seed=42)
        assert a == b

    def test_invalid_fractions_raise(self):
        dates = {"20090401": (5, 50), "20090402": (5, 50), "20090403": (5, 50)}
        files, labels = _make_dated_data(dates)
        with pytest.raises(ValueError):
            day_stratified_split_3way(files, labels, val_fraction=0.0, test_fraction=0.15, seed=0)
        with pytest.raises(ValueError):
            day_stratified_split_3way(files, labels, val_fraction=0.5, test_fraction=0.5, seed=0)
