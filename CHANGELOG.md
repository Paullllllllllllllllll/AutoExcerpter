# Changelog

- **v2.5.0** (13 October 2026) -- Model capabilities follow the vendors'
  documentation, and requests change with them. AutoExcerpter, ChronoTranscriber
  and ChronoMiner now share one capability table, the vendored
  `common/capability_table.py`, which replaces `llm/capabilities.py`'s rules;
  ids lose a `provider:` or Google `models/` prefix before they are matched, as
  in the sibling tools. OpenAI: `flex` is sent only to the models OpenAI lists
  for Flex processing (gpt-4o, the codex, chat-latest and gpt-5.2-pro models
  get no tier); an effort a model does not document becomes the next one it
  does, with one warning per run (`none` and `minimal` give `low` on
  gpt-6-astra and gpt-6.1-sol, `minimal` gives `low` on GPT-6 Luna and Sol and
  GPT-5.6, `max` gives `xhigh` on gpt-5.5 and gpt-5.4, `none` gives `minimal`
  on gpt-5, the pro models take `medium` to `xhigh`); o1 and o1-mini get no
  effort; the chat-latest models are non-reasoning models with 128,000 tokens
  of context and no effort, verbosity or samplers; the codex models accept
  images and take `low` to `xhigh`; the pro and codex models get no verbosity;
  gpt-5.4-mini and gpt-5.4-nano take `original` detail; gpt-5.2-pro and
  gpt-5.4-pro use prompted JSON; o1, o1-pro and o3 use the enforced schema;
  gpt-4.1-nano gets 32,768 and gpt-4 8,192 output tokens; `gpt-5.3` and
  `gpt-5.3-chat`, which OpenAI does not document, take the conservative
  profile. Anthropic: retired Claude 3 and 4.0/4.1 models use prompted JSON,
  Claude Sonnet 4 thinks with 64,000 output tokens, Claude Opus 4 and 4.1 get
  32,000, and Claude Sonnet 4.5 has 200,000 tokens of context. Google: Gemini
  `minimal` becomes `low` on Gemini 3.1 Pro and 3.7/3.8 Flash; the Gemini image
  models get no thinking level and use prompted JSON; Gemini 3.5 Flash, routed
  to 3.6 Flash, gets no samplers; Gemma 4 takes images and a `high` or
  `minimal` thinking level. OpenRouter: anthropic/claude Opus and Haiku 5.5 and
  the other Claude models that reject samplers get none, deepseek-r1 keeps its
  samplers, openai/o3-mini takes no images, and openai/gpt-6 ids are reasoning
  models without samplers. Z.ai GLM ids called directly use JSON mode without
  penalties; `glm-flash-latest` and `glm-5.3-flash:batch` take the conservative
  profile. Image settings recorded for gpt-5.4-mini or gpt-5.4-nano at
  `original` detail, and for `gpt-5.3`, now differ; as with any change of image
  settings, resume refuses such items while pages remain, until `--force`.
- **v2.4.1** (13 October 2026) -- A refactor with no change in behavior. The image
  presets (the recommended values the wizard shows and the request settings of
  each model family) and the building of provider request options move to the
  vendored `common/image_presets.py` and `common/request_options.py`, which
  ChronoTranscriber and ChronoMiner share. AutoExcerpter passes its own values and
  rules (`IMAGE_PRESETS` in `spec.py`, the tables in `llm/options.py`), so
  requests, batch bodies, image settings and the stored image-settings records
  stay as they were; new tests compare them with golden files over a matrix of
  providers, models and options, and resume an item stored by v2.4.0.
- **v2.4.0** (13 October 2026) -- `run` and `batch check --wait` write their
  progress to the file named in `CHRONO_HEARTBEAT_FILE`, so a job launcher can
  show it: the current item's pages processed and its position, an ETA from the
  page rate during the last item, and the batch submissions a wait has seen
  finish. Each write replaces the file in one step, at most every 15 seconds, and
  a write that fails is skipped without affecting the run. The default of
  `--wait-hours` rises from 50 to 98, two rounds of Google's 48-hour batch
  window; the limit stays soft, since a check fails an expired batch and the wait
  then ends on its own.
- **v2.3.1** (13 October 2026) -- The lease heartbeat no longer reports a
  document lease that the run has just released as lost. A release now drops the
  lease from the held leases before its database write, and the heartbeat looks
  each lease up again right before queueing its renewal, so a lease whose release
  began while the heartbeat was renewing is skipped rather than logged as expired.
- **v2.3.0** (13 October 2026) -- `batch check --wait` repeats the check every
  `--poll-minutes` (default 15) until no batch work it can advance is pending,
  so one command collects the transcriptions, submits the summaries, collects
  them and writes the files; no check starts after `--wait-hours` (default 50),
  but a running one finishes. A failure does not end the wait, its counts are
  read from the databases at the end for every submission and document a check
  saw, and it also exits 1 when the deadline passed with work pending or a
  document needs a run, whose `run --batch` command the end line and the new
  JSON fields `needs_run` and `rerun` name; `polls`, `waited_seconds` and
  `stopped` join the JSON line. The batch clients are built once per wait and
  closed at its end. The loop is the new vendored `common/polling.py`, shared
  with ChronoTranscriber and ChronoMiner. A one-pass check is unchanged, except
  that a document whose pages wait for a run is reported as `needs a run`.

- **v2.2.0** (13 October 2026) -- Batch mode: `run --batch` submits each item's
  pages to the batch APIs of OpenAI, Anthropic and Google instead of calling the
  models, and the new `batch check` collects finished batches, submits the
  summaries of the pages they transcribed and writes the files of every document
  left with nothing to do, following the run that submitted it; `batch cancel`
  cancels batches by ID, every unfinished batch, or the open submissions below a
  folder. The request bodies carry what a direct call sends, results are read as
  direct answers are, every result line is stored with its submission, and
  unresolved creates are reconciled with the provider's batch list or settled
  with `--adopt` and `--abandon`. A rerun retries failed pages, with or without
  `--batch`. The guided UI offers `--batch` when every phase's provider has a
  batch API and gains the batch check and cancel tasks; the run summary reports
  `items_submitted`. The store's schema moves to version 2 with a `batch_jobs`
  table, which AutoExcerpter 2.1 refuses as newer. The shared batch backends join
  the vendored `common/` modules. Large folders plan and render faster: inputs are
  hashed on a thread pool, LangChain is imported when a model call needs it,
  native page DPI is found without hashing every image, citations are merged and
  rendered with less work, and OpenAlex DOI lookups go out in batches on four
  workers. gpt-6-luna pages sent at original detail are limited to 3,000 px on
  the long side (`images.original_max_side_px` sets the limit for every OpenAI
  model), and Anthropic structured output keeps nullable fields.
- **v2.1.1** (12 October 2026) -- A resume with a transcription model from
  another provider family, for example after a provider's content filter refused
  some pages, now transcribes only the missing pages instead of skipping the item
  for changed image settings: the stored settings are compared with what the run's
  options give the stored model, so only image settings the user changed stop the
  item. The GLM vision models on OpenRouter (glm-5.3-flash and its siblings, glm-5v,
  glm-4.6v and glm-4.5v) get capability profiles with image input and, where the
  endpoint offers it, structured output.
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
  newline and Word files have fixed archive timestamps. The new `render` command
  writes the files again without model calls and keeps files changed outside the
  tool unless `--force`; the new `migrate` command converts version 1.0
  databases, imports their working logs and registers the existing files,
  consolidates citations from the OpenAlex cache without requests and verifies
  the import; a file that differs from its rendering fails the verification and
  is kept by `render` unless `--force`. Outside code reads the store through the
  views `page_text`, `page_summaries`, `document_citations`,
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
