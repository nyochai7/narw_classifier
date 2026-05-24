import pytest

from narw_classifier.dataset.splits import day_stratified_split_3way, stratified_split_3way


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

    def test_unparseable_filename_raises(self):
        with pytest.raises(ValueError, match="parse date"):
            day_stratified_split_3way(
                ["bogus.aif"], [1], val_fraction=0.15, test_fraction=0.15, seed=0
            )
