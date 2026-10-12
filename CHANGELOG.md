# Changelog

- **v2.1.0** (12 October 2026) -- The transcription and summary rules move into
  the field descriptions of the JSON schemas, written in plain wording, and each
  system prompt becomes one short frame for all providers that points to the
  schema and adds the purpose, the input conventions, the topic context and the
  language rule. The transcription schema drops `image_analysis` (stored answers
  that still carry it load as before) and marks unreadable text as `[illegible]`
  and the uncertain part of a word as `[?]`. Summaries become shorter: two to six
  bullets of 10 to 25 words per page, at most about a quarter of the page's text.
  Anthropic calls cache the system prompt and schema, and the first page of each
  item runs alone so the other pages read the cache. The default transcription
  verbosity is now high.
- **v2.0.0** (11 October 2026) -- Major release: the output folder's
  `autoexcerpter.sqlite` becomes the single source of truth of every run, and the
  transcription and summary files become views rendered from it. Each page's
  transcription and summary commit as units the moment they arrive, every
  provider attempt is stored with the response body as received over HTTP, and
  runs lease the documents they work on, so two runs on one folder never write the
  same document and a reader holding the database delays writes, failing them
  only once it outlasts the 10-minute lease. Resume reads
  only the store and adds the states `legacy`, `blocked` and `refused`; a model
  switch resumes with a note, and a changed input is processed again. The working
  logs in `X_autoexcerpter/`, the `sqlite` format and `--keep-working-files` are
  gone. Rendered files are reproducible: the header shows the time the document
  last changed, in UTC, instead of the elapsed time, text files end with one
  newline and Word files have fixed archive timestamps. The new `render` command writes the
  files again without model calls and keeps files changed outside the tool unless
  `--force`; the new `migrate` command converts version 1.0 databases, imports
  their working logs and registers the existing files, consolidates citations from
  the OpenAlex cache without requests and verifies the import; a file that
  differs from its rendering fails the verification and is kept by `render`
  unless `--force`. Outside code reads
  the store through the views `page_text`, `page_summaries`, `document_citations`,
  `document_status` and `page_attempts`. The guided UI asks for the task first and
  offers render and migrate beside a new run.
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
