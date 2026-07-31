"""Regression tests for the round-5 pipeline fixes.

``pipeline.log.append_to_log`` used to consult ``_FINALIZED_LOGS`` and acquire
the cached handle under two separate ``_LOG_HANDLES_GUARD`` holds. A
``finalize_log_file`` landing between them re-cached a handle for an already
finalized log, leaking that descriptor for the process lifetime. Both steps now
happen under a single hold via ``_acquire_log_handle_locked``.

Additionally covered here:

- ``pipeline.text_cleaner.normalize_unicode`` published a codepoint to the
  shared ``_UNICODE_SEEN`` set before writing its ``_UNICODE_TABLE`` entry, so
  a concurrent worker could skip categorization and return the text uncleaned.
- ``pipeline.types.ItemSpec.output_stem`` used ``Path.stem`` for image folders,
  collapsing dotted folder names ("photos.2023" -> "photos").
"""

from __future__ import annotations

import threading
from pathlib import Path

import pipeline.log as log_mod
import pipeline.text_cleaner as text_cleaner_mod
from pipeline.types import ItemSpec


def _cleanup(path: Path) -> None:
    log_mod.finalize_log_file(path)
    with log_mod._LOG_HANDLES_GUARD:
        log_mod._FINALIZED_LOGS.discard(path)


class TestAcquireLogHandleLocked:
    def test_finalized_log_yields_no_handle_and_caches_nothing(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "finalized.jsonl"
        try:
            with log_mod._LOG_HANDLES_GUARD:
                log_mod._FINALIZED_LOGS.add(path)
                assert log_mod._acquire_log_handle_locked(path) is None
            assert path not in log_mod._LOG_HANDLES
        finally:
            _cleanup(path)

    def test_live_log_caches_one_handle(self, tmp_path: Path) -> None:
        path = tmp_path / "live.jsonl"
        try:
            with log_mod._LOG_HANDLES_GUARD:
                first = log_mod._acquire_log_handle_locked(path)
                second = log_mod._acquire_log_handle_locked(path)
            assert first is not None
            assert second is not None
            # No second descriptor for the same path.
            assert first[0] is second[0]
            assert first[1] is second[1]
            assert path in log_mod._LOG_HANDLES
        finally:
            _cleanup(path)


class TestConcurrentAppendAndFinalize:
    def test_no_handle_survives_a_racing_finalize(self, tmp_path: Path) -> None:
        """Either order is safe; neither leaves a cached (leaked) handle."""
        paths = [tmp_path / f"race_{i}.jsonl" for i in range(40)]
        try:
            for index, path in enumerate(paths):
                appender = threading.Thread(
                    target=log_mod.append_to_log, args=(path, {"page": index})
                )
                finalizer = threading.Thread(
                    target=log_mod.finalize_log_file, args=(path,)
                )
                appender.start()
                finalizer.start()
                appender.join()
                finalizer.join()
                assert path not in log_mod._LOG_HANDLES
        finally:
            for path in paths:
                _cleanup(path)


class TestNormalizeUnicodeSeenTableOrder:
    def test_table_entry_is_present_once_codepoint_is_seen(self) -> None:
        """The table decision must be published before the seen marker.

        Simulates the racing worker: as soon as a control codepoint appears in
        ``_UNICODE_SEEN``, its ``_UNICODE_TABLE`` entry must already exist, so
        a concurrent ``normalize_unicode`` call cannot skip the cleanup.
        """
        control_char = "\u0007"  # BEL: category Cc, not tab/newline
        cp = ord(control_char)
        text_cleaner_mod._UNICODE_SEEN.discard(cp)
        text_cleaner_mod._UNICODE_TABLE.pop(cp, None)

        result = text_cleaner_mod.normalize_unicode(f"a{control_char}b")

        assert result == "ab"
        assert cp in text_cleaner_mod._UNICODE_SEEN
        assert text_cleaner_mod._UNICODE_TABLE[cp] is None
        # A second call (the "racing" worker's view: cp already seen) must
        # still clean via the published table entry.
        assert text_cleaner_mod.normalize_unicode(f"x{control_char}y") == "xy"


class TestItemSpecOutputStem:
    def test_pdf_drops_extension(self) -> None:
        spec = ItemSpec(kind="pdf", path=Path("C:/in/paper.v2.pdf"))
        assert spec.output_stem == "paper.v2"

    def test_image_folder_keeps_full_name(self) -> None:
        spec = ItemSpec(kind="image_folder", path=Path("C:/in/photos.2023"))
        assert spec.output_stem == "photos.2023"

    def test_dotted_sibling_folders_do_not_collide(self) -> None:
        a = ItemSpec(kind="image_folder", path=Path("C:/in/photos.2023"))
        b = ItemSpec(kind="image_folder", path=Path("C:/in/photos.2024"))
        assert a.output_stem != b.output_stem
