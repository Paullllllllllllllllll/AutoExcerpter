# Changelog

- **v1.0.2** (7 October 2026) -- Patch release. Output limits follow the vendor
  documentation: 16,384 tokens for the GPT-5 to GPT-5.3 chat-latest models, 32,768
  for the Gemini 3 Pro and 3.1 Flash image models and gpt-4.1-mini, and 65,536 for
  gemini-2.5-flash-lite; gpt-6.1-sol joins with the gpt-6-sol profile. A resume
  that transcribes the remaining pages with another model no longer stops the item:
  the model and the settings derived from it are not compared, the run warns and
  notes the switch, and each page record names its model, which the database's
  `pages.model` column takes up. Changed user image settings still stop the item,
  and existing working logs resume as before. `original` image detail without a
  model cap now sends pages at their rendered size instead of fitting them into
  the high-detail box.
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
