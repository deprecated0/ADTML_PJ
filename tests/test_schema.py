from src.schema import build_suite


def test_suite_has_expectations():
    suite = build_suite()
    assert len(suite.expectations) >= 5


def test_bad_target_value():
    """GE suite должен ловить невалидный таргет."""
    suite = build_suite()
    types = {e.expectation_type for e in suite.expectations}
    assert "expect_column_values_to_be_in_set" in types
