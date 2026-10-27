import logging
import math
import pytest
from app import create_app
from app.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    validate_confidence_threshold,
)
from app.services.predictor import PredictionService


class TestValidateConfidenceThreshold:
    """Unit tests for the validate_confidence_threshold validator helper."""

    def test_missing_value_returns_default(self):
        """None returns the documented default 0.40 without warning."""
        assert validate_confidence_threshold(None) == DEFAULT_CONFIDENCE_THRESHOLD
        assert validate_confidence_threshold(None) == 0.40

    def test_valid_numeric_string_accepted(self):
        """Valid numeric strings like '0.75' parse to 0.75."""
        assert validate_confidence_threshold("0.75") == 0.75

    def test_valid_boundary_values_accepted(self):
        """Inclusive boundary values '0.0' and '1.0' (and numeric 0.0, 1.0) are accepted."""
        assert validate_confidence_threshold("0.0") == 0.0
        assert validate_confidence_threshold("1.0") == 1.0
        assert validate_confidence_threshold(0.0) == 0.0
        assert validate_confidence_threshold(1.0) == 1.0

    def test_empty_string_falls_back_with_warning(self, caplog):
        """Empty string falls back to 0.40 and logs a warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_confidence_threshold("")
        assert result == DEFAULT_CONFIDENCE_THRESHOLD
        assert "Empty or whitespace-only" in caplog.text

    def test_whitespace_only_string_falls_back_with_warning(self, caplog):
        """Whitespace-only string falls back to 0.40 and logs a warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_confidence_threshold("   \t  \n ")
        assert result == DEFAULT_CONFIDENCE_THRESHOLD
        assert "Empty or whitespace-only" in caplog.text

    def test_non_numeric_string_falls_back_with_warning(self, caplog):
        """Non-numeric string like 'abc' or '0.40%' falls back to 0.40 and logs a warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_confidence_threshold("abc")
        assert result == DEFAULT_CONFIDENCE_THRESHOLD
        assert "Invalid non-numeric" in caplog.text

        with caplog.at_level(logging.WARNING):
            result_pct = validate_confidence_threshold("0.40%")
        assert result_pct == DEFAULT_CONFIDENCE_THRESHOLD

    def test_nan_and_infinity_fall_back_with_warning(self, caplog):
        """NaN, Infinity, and -Infinity (string and float) fall back to 0.40 with warnings."""
        for val in ("NaN", "nan", float("nan"), "Infinity", "-Infinity", float("inf"), float("-inf")):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                result = validate_confidence_threshold(val)
            assert result == DEFAULT_CONFIDENCE_THRESHOLD
            assert "Non-finite" in caplog.text

    def test_negative_and_above_range_fall_back_with_warning(self, caplog):
        """Out-of-range values like '-0.5' and '1.5' fall back to 0.40 with warnings."""
        for val in ("-0.5", -0.5, "-0.0001", "1.5", 1.5, "1.0001"):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                result = validate_confidence_threshold(val)
            assert result == DEFAULT_CONFIDENCE_THRESHOLD
            assert "out of valid range" in caplog.text


class TestConfigAndServiceIntegration:
    """Integration tests verifying config and PredictionService threshold consistency."""

    def test_app_config_uses_validated_env_threshold(self, monkeypatch):
        """App created with valid env threshold loads it into app.config."""
        monkeypatch.setenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.65")
        val = validate_confidence_threshold("0.65")
        assert val == 0.65

    def test_service_uses_app_config_in_active_context(self, app):
        """PredictionService inside active Flask context aligns with app.config."""
        app.config["PREDICTION_CONFIDENCE_THRESHOLD"] = 0.55
        with app.app_context():
            service = PredictionService(lazy_load=True)
            assert service.confidence_threshold == 0.55

    def test_service_explicit_valid_override(self):
        """Explicit valid float or string overrides work on PredictionService."""
        service1 = PredictionService(lazy_load=True, confidence_threshold=0.85)
        assert service1.confidence_threshold == 0.85

        service2 = PredictionService(lazy_load=True, confidence_threshold="0.70")
        assert service2.confidence_threshold == 0.70

        service3 = PredictionService(lazy_load=True, confidence_threshold=0.0)
        assert service3.confidence_threshold == 0.0

        service4 = PredictionService(lazy_load=True, confidence_threshold=1.0)
        assert service4.confidence_threshold == 1.0

    def test_service_explicit_invalid_override_falls_back_safely(self, caplog):
        """Explicit invalid overrides (negative, >1, NaN, non-numeric) safely fall back to 0.40."""
        with caplog.at_level(logging.WARNING):
            service_neg = PredictionService(lazy_load=True, confidence_threshold=-0.5)
            assert service_neg.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
            assert "explicit confidence_threshold" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            service_high = PredictionService(lazy_load=True, confidence_threshold=1.5)
            assert service_high.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            service_nan = PredictionService(lazy_load=True, confidence_threshold=float("nan"))
            assert service_nan.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            service_bad_str = PredictionService(lazy_load=True, confidence_threshold="not_a_number")
            assert service_bad_str.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
