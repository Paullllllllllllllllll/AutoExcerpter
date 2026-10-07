# Changelog

- **v1.0.1** (7 October 2026) -- Documentation release. The README now lists a
  working log that belongs to another input of the same name among the exit-2
  errors, explains in Resume how a plan stops on such a log, and describes in the
  Output Contract how page numbers are corrected within runs of consecutive
  numbers while an excerpt that skips pages keeps its jumps.
- **v1.0.0** (7 October 2026) -- Baseline release. One `autoexcerpter` package
  with two commands built from a single option table: `run` for scripts and
  agents, with `--dry-run` and a one-line `--json` summary, and `ui`, a guided
  terminal run that prints the equivalent command. A settings file holds machine
  values and an optional `defaults:` block. Transcriptions are written as
  Markdown or text, summaries as Markdown and Word files, and every output folder
  gets an `autoexcerpter.sqlite` database with documents, pages, summaries and
  citations. Response formats `schema`, `json` and `text` work on every provider;
  OpenAlex lookups end in explicit outcomes with a persistent cache; runs resume
  from versioned JSONL working logs, and a missing API key stops a run before any
  work.
