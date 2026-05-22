from narw_classifier.utils.filenames import parse_filename


class TestParseFilename:
    def test_train_positive(self):
        parsed = parse_filename("20090328_000000_189s6ms_TRAIN16_1.aif")
        assert parsed is not None
        assert parsed.date == "20090328"
        assert parsed.time == "000000"
        assert parsed.offset == "189s6ms"
        assert parsed.split == "TRAIN"
        assert parsed.index == 16
        assert parsed.label == 1

    def test_train_negative(self):
        parsed = parse_filename("20090328_000000_002s3ms_TRAIN0_0.aif")
        assert parsed is not None
        assert parsed.label == 0
        assert parsed.split == "TRAIN"
        assert parsed.index == 0

    def test_test_no_label(self):
        parsed = parse_filename("20090404_000000_012s0ms_Test0.aif")
        assert parsed is not None
        assert parsed.split == "Test"
        assert parsed.index == 0
        assert parsed.label is None

    def test_unparseable_returns_none(self):
        assert parse_filename("not_a_clip.aif") is None
        assert parse_filename("20090328_000000_002s3ms_TRAIN0_0.wav") is None
        assert parse_filename("") is None

    def test_label_must_be_single_digit(self):
        # Two-digit label would not match the (?P<label>\d) capture group.
        assert parse_filename("20090328_000000_002s3ms_TRAIN0_42.aif") is None
