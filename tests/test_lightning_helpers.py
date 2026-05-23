import pytest

from narw_classifier.utils.lightning import format_lr


class TestFormatLR:
    @pytest.mark.parametrize(
        "lr, expected",
        [
            (1e-3, "1e-3"),
            (3e-4, "3e-4"),
            (1e-5, "1e-5"),
            (3.5e-4, "3.5e-4"),
            (1.0, "1e0"),
            (2.5e-2, "2.5e-2"),
        ],
    )
    def test_compact_scientific(self, lr, expected):
        assert format_lr(lr) == expected
