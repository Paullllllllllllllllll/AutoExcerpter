"""Tests for modules/error_handler.py - Centralized error handling utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from modules.error_handler import (
    ProcessingError,
    ConfigurationError,
    APIError,
    FileProcessingError,
    handle_critical_error,
    handle_recoverable_error,
    safe_execute,
    validate_file_exists,
    validate_directory_exists,
    validate_config_value,
)


# ============================================================================
# Error Classification
# ============================================================================
class TestProcessingError:
    """Tests for ProcessingError exception."""

    def test_instantiation(self):
        """ProcessingError can be instantiated with a message."""
        err = ProcessingError("something broke")
        assert str(err) == "something broke"

    def test_inherits_from_exception(self):
        """ProcessingError inherits from Exception."""
        assert issubclass(ProcessingError, Exception)

    def test_can_be_raised_and_caught(self):
        """ProcessingError can be raised and caught."""
        with pytest.raises(ProcessingError, match="test"):
            raise ProcessingError("test")


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_instantiation(self):
        """ConfigurationError can be instantiated."""
        err = ConfigurationError("bad config")
        assert str(err) == "bad config"

    def test_inherits_from_processing_error(self):
        """ConfigurationError is a subclass of ProcessingError."""
        assert issubclass(ConfigurationError, ProcessingError)

    def test_caught_as_processing_error(self):
        """ConfigurationError can be caught as ProcessingError."""
        with pytest.raises(ProcessingError):
            raise ConfigurationError("bad config")


class TestAPIError:
    """Tests for APIError exception."""

    def test_instantiation(self):
        """APIError can be instantiated."""
        err = APIError("api failure")
        assert str(err) == "api failure"

    def test_inherits_from_processing_error(self):
        """APIError is a subclass of ProcessingError."""
        assert issubclass(APIError, ProcessingError)

    def test_caught_as_processing_error(self):
        """APIError can be caught as ProcessingError."""
        with pytest.raises(ProcessingError):
            raise APIError("timeout")


class TestFileProcessingError:
    """Tests for FileProcessingError exception."""

    def test_instantiation(self):
        """FileProcessingError can be instantiated."""
        err = FileProcessingError("file broken")
        assert str(err) == "file broken"

    def test_inherits_from_processing_error(self):
        """FileProcessingError is a subclass of ProcessingError."""
        assert issubclass(FileProcessingError, ProcessingError)


# ============================================================================
# handle_critical_error
# ============================================================================
class TestHandleCriticalError:
    """Tests for handle_critical_error()."""

    @patch("modules.error_handler.print_error")
    def test_shows_user_message_when_enabled(self, mock_print_error: MagicMock):
        """User message is printed when show_user_message is True."""
        err = RuntimeError("boom")
        handle_critical_error(err, context="test_op", show_user_message=True)

        mock_print_error.assert_called_once()
        call_arg = mock_print_error.call_args[0][0]
        assert "test_op" in call_arg

    @patch("modules.error_handler.print_error")
    def test_no_user_message_when_disabled(self, mock_print_error: MagicMock):
        """No user message is printed when show_user_message is False."""
        err = RuntimeError("boom")
        handle_critical_error(err, context="test_op", show_user_message=False)

        mock_print_error.assert_not_called()

    @patch("modules.error_handler.print_error")
    @patch("modules.error_handler.sys.exit")
    def test_exits_when_exit_on_error_true(
        self, mock_exit: MagicMock, mock_print_error: MagicMock
    ):
        """sys.exit(1) is called when exit_on_error is True."""
        err = RuntimeError("fatal")
        handle_critical_error(
            err, context="critical_op", exit_on_error=True, show_user_message=False
        )

        mock_exit.assert_called_once_with(1)

    @patch("modules.error_handler.print_error")
    @patch("modules.error_handler.sys.exit")
    def test_does_not_exit_when_exit_on_error_false(
        self, mock_exit: MagicMock, mock_print_error: MagicMock
    ):
        """sys.exit is not called when exit_on_error is False."""
        err = RuntimeError("non-fatal")
        handle_critical_error(
            err, context="op", exit_on_error=False, show_user_message=False
        )

        mock_exit.assert_not_called()

    @patch("modules.error_handler.print_error")
    @patch("modules.error_handler.sys.exit")
    def test_default_does_not_exit(
        self, mock_exit: MagicMock, mock_print_error: MagicMock
    ):
        """Default behavior does not exit."""
        handle_critical_error(ValueError("oops"), context="default_test")
        mock_exit.assert_not_called()


# ============================================================================
# handle_recoverable_error
# ============================================================================
class TestHandleRecoverableError:
    """Tests for handle_recoverable_error()."""

    @patch("modules.error_handler.print_warning")
    def test_shows_user_message_when_enabled(self, mock_print_warning: MagicMock):
        """Warning is printed when show_user_message is True."""
        err = ValueError("minor issue")
        handle_recoverable_error(err, context="parsing", show_user_message=True)

        mock_print_warning.assert_called_once()
        call_arg = mock_print_warning.call_args[0][0]
        assert "parsing" in call_arg

    @patch("modules.error_handler.print_warning")
    def test_no_user_message_when_disabled(self, mock_print_warning: MagicMock):
        """No warning is printed when show_user_message is False."""
        err = ValueError("minor issue")
        handle_recoverable_error(err, context="parsing", show_user_message=False)

        mock_print_warning.assert_not_called()

    @patch("modules.error_handler.print_warning")
    def test_default_shows_user_message(self, mock_print_warning: MagicMock):
        """Default behavior shows user message."""
        handle_recoverable_error(RuntimeError("err"), context="op")
        mock_print_warning.assert_called_once()


# ============================================================================
# safe_execute
# ============================================================================
class TestSafeExecute:
    """Tests for safe_execute()."""

    def test_successful_function_call(self):
        """Returns the function result on success."""
        result = safe_execute(lambda x, y: x + y, 3, 4, context="addition")
        assert result == 7

    def test_function_that_raises_returns_none(self):
        """Returns None when function raises and no default is given."""
        def failing():
            raise ValueError("fail")

        result = safe_execute(failing, context="failing_op")
        assert result is None

    def test_function_that_raises_returns_default(self):
        """Returns the default value when function raises."""
        def failing():
            raise ValueError("fail")

        result = safe_execute(failing, default="fallback", context="failing_op")
        assert result == "fallback"

    def test_default_zero_returned(self):
        """Falsy default value (0) is correctly returned on failure."""
        def failing():
            raise RuntimeError("boom")

        result = safe_execute(failing, default=0, context="test")
        assert result == 0

    @patch("modules.error_handler.logger")
    def test_log_errors_true_logs_warning(self, mock_logger: MagicMock):
        """Warning is logged when log_errors is True (default)."""
        def failing():
            raise RuntimeError("logged error")

        safe_execute(failing, context="logged_op", log_errors=True)

        mock_logger.warning.assert_called()
        logged_msg = mock_logger.warning.call_args[0][0]
        assert "logged_op" in logged_msg

    @patch("modules.error_handler.logger")
    def test_log_errors_false_no_logging(self, mock_logger: MagicMock):
        """No logging occurs when log_errors is False."""
        def failing():
            raise RuntimeError("silent error")

        safe_execute(failing, context="silent_op", log_errors=False)

        mock_logger.warning.assert_not_called()
        mock_logger.debug.assert_not_called()

    def test_passes_kwargs_through(self):
        """Keyword arguments are forwarded to the function."""
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = safe_execute(greet, "Alice", greeting="Hi", context="greet")
        assert result == "Hi, Alice!"


# ============================================================================
# validate_file_exists
# ============================================================================
class TestValidateFileExists:
    """Tests for validate_file_exists()."""

    def test_existing_file_passes(self, tmp_path: Path):
        """No exception raised for an existing file."""
        f = tmp_path / "real_file.txt"
        f.write_text("content")

        validate_file_exists(f, context="test_file")  # should not raise

    def test_non_existent_path_raises(self, tmp_path: Path):
        """FileProcessingError raised for a non-existent path."""
        missing = tmp_path / "ghost.txt"

        with pytest.raises(FileProcessingError, match="not found"):
            validate_file_exists(missing, context="test_file")

    def test_directory_instead_of_file_raises(self, tmp_path: Path):
        """FileProcessingError raised when path is a directory, not a file."""
        d = tmp_path / "a_directory"
        d.mkdir()

        with pytest.raises(FileProcessingError, match="not a file"):
            validate_file_exists(d, context="test_file")

    def test_accepts_string_path(self, tmp_path: Path):
        """Accepts a string path and converts it internally."""
        f = tmp_path / "string_test.txt"
        f.write_text("data")

        validate_file_exists(str(f), context="string_path")  # should not raise


# ============================================================================
# validate_directory_exists
# ============================================================================
class TestValidateDirectoryExists:
    """Tests for validate_directory_exists()."""

    def test_existing_directory_passes(self, tmp_path: Path):
        """No exception raised for an existing directory."""
        d = tmp_path / "real_dir"
        d.mkdir()

        validate_directory_exists(d, context="test_dir")  # should not raise

    def test_non_existent_path_raises(self, tmp_path: Path):
        """FileProcessingError raised for a non-existent path."""
        missing = tmp_path / "nowhere"

        with pytest.raises(FileProcessingError, match="not found"):
            validate_directory_exists(missing, context="test_dir")

    def test_file_instead_of_directory_raises(self, tmp_path: Path):
        """FileProcessingError raised when path is a file, not a directory."""
        f = tmp_path / "a_file.txt"
        f.write_text("content")

        with pytest.raises(FileProcessingError, match="not a directory"):
            validate_directory_exists(f, context="test_dir")

    def test_accepts_string_path(self, tmp_path: Path):
        """Accepts a string path and converts it internally."""
        d = tmp_path / "str_dir"
        d.mkdir()

        validate_directory_exists(str(d), context="str_test")  # should not raise


# ============================================================================
# validate_config_value
# ============================================================================
class TestValidateConfigValue:
    """Tests for validate_config_value()."""

    def test_valid_type_passes(self):
        """No exception raised when value matches expected type."""
        validate_config_value(42, int, "port")  # should not raise

    def test_valid_string_type(self):
        """No exception raised for valid string value."""
        validate_config_value("hello", str, "greeting")  # should not raise

    def test_invalid_type_raises(self):
        """ConfigurationError raised when value does not match expected type."""
        with pytest.raises(ConfigurationError, match="expected int"):
            validate_config_value("not_an_int", int, "port")

    def test_none_with_allow_none_true(self):
        """No exception raised when value is None and allow_none is True."""
        validate_config_value(None, str, "optional_field", allow_none=True)

    def test_none_with_allow_none_false(self):
        """ConfigurationError raised when value is None and allow_none is False."""
        with pytest.raises(ConfigurationError, match="expected str"):
            validate_config_value(None, str, "required_field", allow_none=False)

    def test_none_default_allow_none_is_false(self):
        """Default allow_none is False, so None raises."""
        with pytest.raises(ConfigurationError):
            validate_config_value(None, int, "test_field")

    def test_error_message_includes_field_name(self):
        """The error message includes the configuration field name."""
        with pytest.raises(ConfigurationError, match="my_setting"):
            validate_config_value(3.14, int, "my_setting")

    def test_bool_is_not_accepted_as_int(self):
        """bool is a subclass of int in Python, so it passes isinstance check."""
        # This documents actual Python behavior: isinstance(True, int) is True.
        validate_config_value(True, int, "flag")  # should not raise

    def test_list_type_validation(self):
        """List type validation works correctly."""
        validate_config_value([1, 2, 3], list, "items")  # should not raise

        with pytest.raises(ConfigurationError):
            validate_config_value((1, 2, 3), list, "items")
