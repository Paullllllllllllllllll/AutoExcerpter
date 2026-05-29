"""Deterministic repair of spurious line-breaks in transcription .txt files.

This package reverses the damage caused by running ``wrap_long_lines`` over
already laid-out transcription text (see ``pipeline/text_cleaner.py``). The
repair only moves whitespace and merges line-break hyphenation; it never adds,
removes, reorders, or re-spells content. A content-signature verifier proves
that guarantee on every file.
"""
