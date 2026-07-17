"""Orchestrate the deterministic line-break repair over a folder of files.

Steps: enumerate transcription ``.txt`` files (those with a ``# Transcription
of:`` header), back them up to a versioned zip, repair each one, verify the
content-preservation gate, write only files that pass, and emit a markdown
report plus a JSON of flagged cases for the read-only LLM audit.

Usage (from the repository root)::

    uv run python -m scripts.repair_layout.run_repair --root "<folder>" \
        [--backup-dir "<folder>\\backup"] [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import zipfile
from pathlib import Path

from scripts.repair_layout.repair import repair_text
from scripts.repair_layout.verifier import VerifyResult, verify

HEADER_MARKER = "# Transcription of:"
_LONG_LINE_AUDIT_THRESHOLD = 110


def find_targets(root: Path) -> list[Path]:
    """Return transcription .txt files under root, by their header marker."""
    targets: list[Path] = []
    for path in sorted(root.rglob("*.txt")):
        if "backup" in {part.lower() for part in path.parts}:
            continue
        try:
            with open(path, encoding="utf-8", newline="") as handle:
                head = handle.read(256)
        except (OSError, UnicodeDecodeError):
            continue
        if head.lstrip("﻿").startswith(HEADER_MARKER):
            targets.append(path)
    return targets


def read_text_preserve(path: Path) -> tuple[str, bool]:
    """Read a file as UTF-8, returning (text_with_lf, had_crlf)."""
    with open(path, encoding="utf-8", newline="") as handle:
        raw = handle.read()
    had_crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return text, had_crlf


def write_text_preserve(path: Path, text: str, had_crlf: bool) -> None:
    """Write text back, preserving the file's original newline style."""
    eol = "\r\n" if had_crlf else "\n"
    data = text.replace("\n", eol)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(data)


def _next_backup_version(backup_dir: Path, stem: str, date: str) -> Path:
    """Return the next non-colliding versioned backup zip path."""
    version = 1
    while True:
        candidate = backup_dir / f"{stem}_{date}_v{version}.zip"
        if not candidate.exists():
            return candidate
        version += 1


def create_backup(targets: list[Path], root: Path, backup_dir: Path, date: str) -> Path:
    """Zip all target files (paths relative to root) and document the snapshot."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    zip_path = _next_backup_version(backup_dir, "literature_transcriptions", date)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in targets:
            archive.write(path, arcname=str(path.relative_to(root)))

    with zipfile.ZipFile(zip_path, "r") as archive:
        entry_count = len(archive.namelist())
    if entry_count != len(targets):
        raise RuntimeError(
            f"Backup verification failed: zipped {entry_count} of {len(targets)} files"
        )

    doc = backup_dir / "backup_doc.md"
    line = (
        f"- `{zip_path.name}` ({date}): {len(targets)} transcription .txt files, "
        f"pre line-break-repair snapshot. Restore by extracting over `{root}`.\n"
    )
    if doc.exists():
        existing = doc.read_text(encoding="utf-8")
        doc.write_text(existing.rstrip("\n") + "\n" + line, encoding="utf-8")
    else:
        doc.write_text("# Backup documentation\n\n" + line, encoding="utf-8")
    return zip_path


@dataclasses.dataclass
class FileResult:
    """Per-file outcome of the repair run."""

    path: str
    changed: bool
    written: bool
    verify: VerifyResult
    kept_hyphens: list[str]
    merged_hyphens: int
    long_lines: list[str]
    width_estimates: list[int]


def process_file(path: Path, root: Path, dry_run: bool) -> FileResult:
    """Repair, verify, and (unless dry-run) write back a single file."""
    text, had_crlf = read_text_preserve(path)
    repaired, audit = repair_text(text)
    changed = repaired != text
    result = verify(text, repaired)

    written = False
    if changed and result.passed and not dry_run:
        write_text_preserve(path, repaired, had_crlf)
        written = True

    # Flag long lines in the repaired output that are newly formed (i.e. not
    # present verbatim in the original text), which merit a manual look.
    original_lines = set(text.split("\n"))
    long_lines = [
        line
        for line in repaired.split("\n")
        if len(line) > _LONG_LINE_AUDIT_THRESHOLD and line not in original_lines
    ]
    kept = [f"{d.left}-{d.right}" for d in audit.hyphen_decisions if d.kept]
    merged = sum(1 for d in audit.hyphen_decisions if not d.kept)

    return FileResult(
        path=str(path.relative_to(root)),
        changed=changed,
        written=written,
        verify=result,
        kept_hyphens=kept,
        merged_hyphens=merged,
        long_lines=long_lines,
        width_estimates=audit.page_width_estimates,
    )


def write_reports(results: list[FileResult], backup_dir: Path, date: str) -> None:
    """Write the human-readable markdown report and the LLM-audit JSON."""
    failures = [r for r in results if r.changed and not r.verify.passed]
    changed = [r for r in results if r.changed]
    flagged = [r for r in results if r.kept_hyphens or r.long_lines]

    lines = [
        f"# Line-break repair report ({date})",
        "",
        f"Files scanned: {len(results)}; changed: {len(changed)}; "
        f"FAILED gate: {len(failures)}.",
        "",
        "All gates: equal content signature, equal ordered page markers, equal "
        "alphanumeric count, and line count never increases.",
        "",
        "## Failures (must be empty)",
        "",
    ]
    if failures:
        for r in failures:
            lines.append(f"- FAIL `{r.path}`: {r.verify.summary()}")
    else:
        lines.append("None. Every changed file preserved content exactly.")
    lines += [
        "",
        "## Per-file",
        "",
        "| File | written | gate | lines | hyphens kept/merged |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        gate = "PASS" if r.verify.passed else "FAIL"
        lines.append(
            f"| {r.path} | {r.written} | {gate} | "
            f"{r.verify.lines_before}->{r.verify.lines_after} | "
            f"{len(r.kept_hyphens)}/{r.merged_hyphens} |"
        )
    (backup_dir / "repair_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    audit_payload = [
        {
            "path": r.path,
            "kept_hyphens": r.kept_hyphens,
            "long_lines": r.long_lines,
        }
        for r in flagged
    ]
    (backup_dir / "repair_audit_flagged.json").write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Repair transcription line breaks.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-backup", action="store_true")
    args = parser.parse_args()

    root: Path = args.root
    backup_dir: Path = args.backup_dir or (root / "backup")
    date = _dt.date.today().strftime("%d_%m_%Y")

    targets = find_targets(root)
    if args.limit is not None:
        targets = targets[: args.limit]
    print(f"Found {len(targets)} transcription .txt files under {root}")

    if not args.dry_run and not args.skip_backup:
        zip_path = create_backup(targets, root, backup_dir, date)
        print(f"Backup written and verified: {zip_path}")

    results = [process_file(path, root, args.dry_run) for path in targets]

    backup_dir.mkdir(parents=True, exist_ok=True)
    write_reports(results, backup_dir, date)

    changed = sum(1 for r in results if r.changed)
    written = sum(1 for r in results if r.written)
    failures = sum(1 for r in results if r.changed and not r.verify.passed)
    print(
        f"Changed: {changed}; written: {written}; gate failures: {failures}; "
        f"dry_run={args.dry_run}. Reports in {backup_dir}"
    )


if __name__ == "__main__":
    main()
