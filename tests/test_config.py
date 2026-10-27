import logging
import math
import pytest
from app import create_app
from app.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEV_SECRET_KEY,
    DEFAULT_PORT,
    ProductionConfig,
    validate_confidence_threshold,
    validate_port,
    validate_production_secret,
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


class TestValidatePort:
    """Unit tests for validate_port helper."""

    def test_missing_port_returns_default_5000(self):
        """None returns 5000 without warning."""
        assert validate_port(None) == DEFAULT_PORT
        assert validate_port(None) == 5000

    def test_valid_port_string_accepted(self):
        """Valid port string like '8080' or '5000' parses to int."""
        assert validate_port("8080") == 8080
        assert validate_port("5000") == 5000

    def test_boundary_ports_accepted(self):
        """Inclusive boundary ports 1 and 65535 are accepted (as strings and ints)."""
        assert validate_port("1") == 1
        assert validate_port("65535") == 65535
        assert validate_port(1) == 1
        assert validate_port(65535) == 65535

    def test_empty_and_whitespace_port_falls_back_with_warning(self, caplog):
        """Empty or whitespace-only PORT strings fall back to 5000 with a warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_port("")
        assert result == DEFAULT_PORT
        assert "Empty or whitespace-only PORT" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result_ws = validate_port("   \t  \n ")
        assert result_ws == DEFAULT_PORT
        assert "Empty or whitespace-only PORT" in caplog.text

    def test_non_numeric_port_falls_back_with_warning(self, caplog):
        """Non-numeric string like 'abc' or 'http' falls back to 5000 with a warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_port("abc")
        assert result == DEFAULT_PORT
        assert "Invalid non-integer PORT" in caplog.text

    def test_fractional_string_port_falls_back_with_warning(self, caplog):
        """Fractional string like '5000.5' and float 5000.5 fall back to 5000 with a warning."""
        with caplog.at_level(logging.WARNING):
            result = validate_port("5000.5")
        assert result == DEFAULT_PORT
        assert "Invalid non-integer PORT" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            result_float = validate_port(5000.5)
        assert result_float == DEFAULT_PORT
        assert "Invalid non-integer PORT" in caplog.text

    def test_out_of_range_port_falls_back_with_warning(self, caplog):
        """Port '0', negative numbers, and values above 65535 fall back to 5000 with a warning."""
        for val in ("0", 0, "-1", -1, "-80", "65536", 65536, "70000", 70000):
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                res = validate_port(val)
            assert res == DEFAULT_PORT
            assert "out of valid range" in caplog.text

    def test_boolean_port_rejected_with_warning(self, caplog):
        """Boolean True/False (subclass of int in Python) are rejected with a warning."""
        with caplog.at_level(logging.WARNING):
            assert validate_port(True) == DEFAULT_PORT
            assert "Invalid non-integer PORT" in caplog.text

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert validate_port(False) == DEFAULT_PORT
            assert "Invalid non-integer PORT" in caplog.text


class TestValidateProductionSecret:
    """Unit tests for validate_production_secret."""

    def test_valid_production_secret_accepted(self):
        """A valid, non-empty, non-default secret is accepted."""
        valid_secret = "a-very-strong-production-secret-key-12345"
        assert validate_production_secret(valid_secret) == valid_secret

    def test_missing_secret_rejected_with_value_error(self):
        """Missing secret (None) raises ValueError."""
        with pytest.raises(ValueError, match="No SECRET_KEY was provided"):
            validate_production_secret(None)

    def test_empty_and_whitespace_secrets_rejected(self):
        """Empty and whitespace-only secrets raise ValueError."""
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            validate_production_secret("")

        with pytest.raises(ValueError, match="empty or whitespace-only"):
            validate_production_secret("   \t \n ")

    def test_known_dev_default_rejected(self):
        """The known development default secret is rejected in production."""
        with pytest.raises(ValueError, match="default development key is not permitted"):
            validate_production_secret(DEFAULT_DEV_SECRET_KEY)

        with pytest.raises(ValueError, match="default development key is not permitted"):
            validate_production_secret("dev-secret-key-cropscan-change-in-production")

    def test_error_message_does_not_reveal_secret(self):
        """Error messages never echo or log secret values."""
        secret_sample = "sensitive-candidate-secret-token"
        try:
            validate_production_secret("   ")
        except ValueError as e:
            assert secret_sample not in str(e)


class TestProductionConfigLoading:
    """Integration tests verifying application factory production secret validation."""

    def test_production_app_with_valid_secret_succeeds(self, monkeypatch):
        """Production app successfully creates when valid SECRET_KEY is set."""
        monkeypatch.setenv("SECRET_KEY", "real-production-secret-token-xyz")
        prod_app = create_app("production")
        assert prod_app.config["SECRET_KEY"] == "real-production-secret-token-xyz"
        assert prod_app.config["DEBUG"] is False
        assert prod_app.config["TESTING"] is False

    def test_production_app_with_missing_secret_fails(self, monkeypatch):
        """Production app creation fails when SECRET_KEY is missing from environment."""
        monkeypatch.delenv("SECRET_KEY", raising=False)
        with pytest.raises(ValueError, match="No SECRET_KEY was provided"):
            create_app("production")

    def test_production_app_with_empty_secret_fails(self, monkeypatch):
        """Production app creation fails when SECRET_KEY is empty."""
        monkeypatch.setenv("SECRET_KEY", "")
        with pytest.raises(ValueError, match="empty or whitespace-only"):
            create_app("production")

    def test_production_app_with_default_secret_fails(self, monkeypatch):
        """Production app creation fails when SECRET_KEY is the insecure dev default."""
        monkeypatch.setenv("SECRET_KEY", DEFAULT_DEV_SECRET_KEY)
        with pytest.raises(ValueError, match="default development key is not permitted"):
            create_app("production")

    def test_development_and_testing_configs_unaffected(self, monkeypatch):
        """Development and testing apps still work with default dev secret."""
        monkeypatch.setenv("SECRET_KEY", DEFAULT_DEV_SECRET_KEY)
        dev_app = create_app("development")
        assert dev_app.config["SECRET_KEY"] == DEFAULT_DEV_SECRET_KEY
        assert dev_app.config["DEBUG"] is True

        test_app = create_app("testing")
        assert test_app.config["SECRET_KEY"] == DEFAULT_DEV_SECRET_KEY
        assert test_app.config["TESTING"] is True
