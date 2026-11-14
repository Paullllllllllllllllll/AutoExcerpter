# Changelog

- **v1.11.1** (14 November 2026) -- The docstring of the batch wait line now
  states only its fixed prefix, `Waiting for batches`, and no longer describes
  how other programs read the log. The line itself and the tool's behavior are
  unchanged.
- **v1.11.0** (14 November 2026) -- The repository ships a skill for coding
  agents in `skills/autoexcerpter/`. `SKILL.md` gives the steps of a job (dry run,
  sample, run, verify), the conventions for outputs, page ranges and context files,
  and the handling of resumed and failed items; `batch.md` covers the two phases of
  a batch job. The skill defers to the README and `--help` for flags and behavior,
  and the README's new Agent Skill section says how to install it. The tool's code
  and behavior are unchanged.
- **v1.10.0** (13 November 2026) -- `run` gains `--open-items N` (default 1),
  which keeps several items open at once. `--concurrency` now bounds the requests
  of the whole run, and `--open-items` sets how many items share them: a further
  item opens whenever the open ones have fewer pages left than `--concurrency`,
  so a folder of short documents keeps every request slot busy. On 40 two-page
  documents against an endpoint answering after 1 s, a run at `--concurrency 16`
  took 87 s with one open item and 16 s with eight. Items store their citations
  and write their files one at a time, and the summary, the exit code and the
  failure list keep the item order whatever the timing. A provider stop opens no
  further item and closes every open item as an interrupted one; Ctrl+C cancels
  them all and leaves no lease held. An item that was only opening when the run
  stopped is left as it was, also under `--force`, and counts as not started.
  The option has no effect with `--batch` or on a single item, and the run warns
  when it is given there. The stored run spec records `run.open_items` only
  above 1. With the default of 1 a run behaves as before. The shared package
  gains `common/scheduler.py`, and the page sources of all PDFs share one lock
  for PyMuPDF.

- **v1.9.0** (13 November 2026) -- AutoExcerpter now reads input folders by the
  rules ChronoMiner and ChronoTranscriber follow. A folder input gives only its
  own PDFs and the image folders that are the folder itself or one of its direct
  subfolders. A run over a deeper tree needs the new `--recursive` flag; without
  it the run warns how many items it left in subfolders. Items are processed and
  numbered in the natural order of their paths below the input folder, PDFs and
  image folders together (`page_2` before `page_10`, letter case ignored).
  `--select N` can therefore name another item than before, and a `--select` name
  search now matches the whole relative path. `--files` takes item paths relative
  to the input folder, with forward or back slashes, at every level of the tree. A
  bare file or folder name still works when exactly one item has it; a name
  several items share is refused with its candidates, and `./name` then selects
  the item in the input folder itself. Two spellings of one path give one item,
  and a path that leaves the input folder is refused. Discovery no longer enters
  `.mabillon` folders or directory links, Windows junctions included. A
  `run --batch` renders and submits page images in slices of at most 64 MiB, so a
  long book becomes several batch submissions and is never held in memory whole.
  The shared package gains `common/discovery.py`. Document names and output paths
  are unchanged, so existing output folders resume as before.

- **v1.8.0** (13 November 2026) -- A run that a provider stops ends with the new
  exit code 3. When the attempt context of `ext.py` answers a failed call with
  the new decision `STOP`, the requests already sent finish and are stored, no
  further unit and no further item starts, the item in work keeps its stored pages
  and writes no file, and its lease is released; the final error line and the new
  `stopped` field of the `--json` line name the reason, `--help` lists the exit
  code, and a rerun resumes with the rest. Planning reads the database of an
  output folder once and classifies all its items from that read, which cut the
  planning of 1,000 stored documents from 13 s to 2.7 s. The stored message of a
  failed attempt names the transport error behind it, for example
  `(cause: httpx.ReadTimeout)`. The shared package gains `Decision.STOP`,
  `RunStopped`, `StopSignal`, `heed_stop`, `ExitCode.STOPPED` and the
  connection-level `resume_state`.
- **v1.7.2** (9 November 2026) -- The tests of the decision parser, the page
  types, the decision images and the stale-unit triggers carry type annotations, so
  `mypy autoexcerpter tests` passes again; no behavior changes.
- **v1.7.1** (8 November 2026) -- pyasn1 moves from 0.6.4 to 0.6.5 in
  `uv.lock` for GHSA-2qm8-x737-gmw5, GHSA-9m75-vc3g-4x39, GHSA-fm9c-5rcf-2q5p
  and GHSA-mm57-4v3p-rchx. No source change.
- **v1.7.0** (8 November 2026) -- Native PDF renders now stay within the
  `max_pixels_per_page` memory guard (24 MP) like numeric renders and record
  `memory_guard` when it lowers the zoom. A shared `check_request_size` in
  `common/provider_requests.py` holds every request to its provider's limits: each
  image, the context image included, within the family's per-image payload limit, and a
  Google request's images and prompt text together within 20 MB; a request over a limit
  is not sent and fails its page or chunk with the reason, while the rest of a batch is
  submitted and requests within the limits are unchanged.
- **v1.6.0** (8 November 2026) -- Gemini models at `--image-detail original` now get
  their pages at the rendered size, without the 768x1536 box and its padding, and every
  image part of synchronous and batch requests carries media resolution `ultra_high`;
  before, Google dropped `original` and sent the box-fitted page at `high`. The shared
  `common/image_presets.py` sets the Google section accordingly and adds
  `native_render_unbounded`, which the local check of native renders now calls;
  ChronoTranscriber, ChronoMiner and AutoExcerpter carry the same copy. The warning that
  a resize profile discards native resolution no longer appears for pages sent unscaled.
- **v1.5.0** (7 November 2026) -- Refreshed dependencies within their current majors:
  `langchain-core` 1.7.6, `langchain-openai` 1.7.4, `langchain-anthropic` 1.7.10,
  `langchain-google-genai` 4.4.3, and the dev tools `ruff` 0.17.4, `pytest-asyncio`
  1.4.1.post1, and `types-PyYAML` 6.0.12.20261103, with their floors raised to match.
  No dependencies were removed. Major releases of `anthropic` (2.6.0), `openai`
  (3.46.0), `google-genai` (3.0.0), and `PyMuPDF` (2.1) are held for a separate
  migration.
- **v1.4.0** (6 November 2026) -- `run` takes `--presence-penalty`,
  `--frequency-penalty`, `--top-k`, `--min-p` and `--repetition-penalty`, each also
  per phase (`--transcription-...`, `--summary-...`) with a fallback to the
  unprefixed flag, settable from `defaults:` like `--temperature`. Unset, a request
  is unchanged. A set value goes to the providers that take it: OpenRouter and
  custom endpoints get all five, Google models that take samplers the penalties and
  `top_k`, Anthropic `top_k` when no thinking is sent; the OpenAI Responses API
  takes none of them. Elsewhere the run warns that the value has no effect, and the
  dry-run view drops it. The shared `common/` modules `provider_requests.py`,
  `spec_view.py` and `standard_options.py` are synced with ChronoMiner and
  ChronoTranscriber.
- **v1.3.0** (6 November 2026) -- `run` takes `--openrouter-provider NAME`, and per
  phase `--transcription-openrouter-provider` and `--summary-openrouter-provider`,
  which fall back to it; the flag pins the provider that serves the phase's
  OpenRouter model requests: each request then carries OpenRouter's routing object
  `{"order": [NAME], "allow_fallbacks": false, "require_parameters": true}` as
  `provider`, so one provider serves the run and a request it cannot serve fails
  instead of going to another provider. The flag is set on the command line only;
  for other providers it has no effect and says so, and the dry-run spec shows it
  for OpenRouter models. Decision-model requests keep OpenRouter's routing. The
  shared `common/` modules `provider_requests.py`, `spec_view.py` and
  `standard_options.py` are synced with ChronoMiner and
  ChronoTranscriber.
- **v1.2.1** (5 November 2026) -- A rerun of `run` after a killed run no longer skips
  the killed run's items for the 10 minutes its leases stay valid. Before the run it
  reads the leases on its items again every 90 seconds; a lease whose expiry does not
  move belongs to a run that has stopped, and the rerun waits until it expires and
  then processes the item, while an item whose lease is renewed stays blocked and
  exits 1. A leased document without stored units is now blocked at planning.
- **v1.2.0** (1 November 2026) -- Refresh the dependencies within their current majors
  (among them `langchain-core`, `langchain-openai` and `langchain-anthropic` to 1.7,
  `langchain-google-genai` to 4.4, `openai` to 2.54, `anthropic` to 0.125,
  `google-genai` to 2.34 and PyMuPDF to 1.28.2) and cap `anthropic`, `openai` and
  PyMuPDF below their next majors. PDF code imports `pymupdf` instead of the deprecated
  `fitz`, whose warning broke the interrupt handshake. A new `error_type_of` in
  `common/capture.py` keeps stored error types at their provider names under the
  LangChain 1.7 error wrappers, so a server error is still recorded as
  `InternalServerError`. The shared capability table adds `gemini-4-argon` and corrects
  the context limit of `gemini-3.8-flash` and `gemini-3.7-flash` to 1,048,576 tokens.
- **v1.1.10** (30 October 2026) -- A complete item no longer takes a requested
  file the store holds no record of as current: such a file is rendered, so the
  overwrite guard records it when it matches the rendering and otherwise keeps
  it with a warning. The vendored common package matches ChronoTranscriber
  v1.2.3.
- **v1.1.9** (30 October 2026) -- The vendored common package matches
  ChronoTranscriber v1.2.2: a write that creates, writes, resets or deletes a
  document's units marks every rendered file of the document stale, and
  rendering clears the flag of each file it records. AutoExcerpter behaves as
  before, since a run already marked an item's files stale when it started to
  work the item.
- **v1.1.8** (30 October 2026) -- The vendored common package matches
  ChronoTranscriber v1.2.0: it gains the classifier file format
  (`classifiers.py`), the shared decision-model call loop
  (`decision_runner.py`), the classification store component
  (`classify_store.py`), a decision-image helper and the classifier options,
  and the decision parser now rejects scores outside their levels and keeps
  the usage of malformed answers. A store migration is refused while another
  run holds a live lease. AutoExcerpter registers no classification component
  and none of its commands uses the new modules, so its behavior is unchanged;
  the tests of the vendored modules come along, with `pytest-asyncio` as a
  development dependency.
- **v1.1.7** (29 October 2026) -- The Pillow requirement is raised to
  13.0, the first release that fixes the FlashPix, TIFF-writer and
  multi-frame loader advisories, so that no install resolves a vulnerable
  Pillow. The lock already held 13.0.0.
- **v1.1.6** (29 October 2026) -- The vendored common package matches
  ChronoTranscriber v1.1.1: it gains `decisions.py` (request bodies and typed
  answers for OpenAI's `/v1/decisions` and OpenRouter's decision API),
  `decision_client.py`, `page_types.py`, the shared decision-model options and
  the `decisions` settings schema. No command uses them yet, so behavior is
  unchanged.
- **v1.1.5** (27 October 2026) -- The capability table knows `gpt-6.1-terra`:
  it takes text and image input and structured outputs like `gpt-6.1-sol`,
  accepts the reasoning efforts none to max, and shares the GPT-6 handling of
  verbosity, Flex and prompt caching. Before, the model resolved as text-only
  and transcription refused it. The vendored `common/` modules now meet the
  Google docstring rules as well, so ruff no longer exempts them. Two timing
  tests, the OpenAlex polite-delay spacing and the page start gate, allow
  wider scheduling delays, since a start recorded late under a parallel test
  run shortens the next measured gap.
- **v1.1.4** (27 October 2026) -- Ruff now checks docstrings with its
  pydocstyle rules under the Google convention, outside the tests and the
  vendored `common/` package. Multi-line summaries become one-line summaries
  followed by a description, the protocol members and `Parser.error` gain
  docstrings, and docstrings that show LaTeX backslashes are raw strings.
  Behavior is unchanged.
- **v1.1.3** (26 October 2026) -- Security patch: the lockfile moves Pillow to
  13.0.0, which closes the Pillow 12.3.0 advisories, among them the TIFF decoder
  reads GHSA-c6hw-5grf-ff96 and GHSA-q45j-58j2-45rg and the MIC frame-count loop
  GHSA-22hg-r758-rrw9, which scanned page images can reach. Dependency versions
  only; no code changes.

- **v1.1.2** (24 October 2026) -- Security patch: the lockfile moves certifi to
  2026.10.23, which closes GHSA-6q48-6c7w-9gw2, and urllib3 to 2.8.0, which closes
  GHSA-8988-9cw3-xx77, GHSA-vxq7-64xx-v4gw and GHSA-gh4c-6fx4-qh6g. Dependency versions
  only; no code changes.

- **v1.1.1** (21 October 2026) -- The longest run of consecutive page numbers
  that anchors a section now holds numbers of one type only. A chapter opening
  read as roman i before arabic 2, 3 and onward no longer joins the arabic run
  and makes the anchor roman, which rendered the whole body of a book in roman
  numerals; roman front matter stays roman and arabic content stays arabic.

- **v1.1.0** (21 October 2026) -- Every page of the transcription file now opens with
  `<page_break pdf="N"/>` (`image="N"` for image folders), so a page is the
  block from its marker to the next; the model's `<page_number>` tags stay where
  the page prints them. The transcription prompt tags only the page's own folio,
  never a chapter, section, figure, table, note or signature number, nor a second
  pagination. The page numbering checks each two-page spread flag against the
  neighboring scans and reads an unsupported one as a single page, numbered from
  the model's second number when only that number continues the previous page. A
  short run of numbers inside another numbering of the block, such as section
  numbers between folios, no longer anchors a section and takes the surrounding
  numbering, while runs after a real skip or repeat keep their own anchors.
- **v1.0.0** (19 October 2026) -- First release. The package `autoexcerpter`
  keeps its core free of terminal input and output: `run.py` holds the entry
  points `discover`, `plan` and `run`, which the command line (`cli/`) and the
  guided UI (`ui/`) both call, and the core reports progress through the run
  events of `events.py`. `spec.py` declares every option once in the option
  table `OPTIONS`, from which the argparse parser and the wizard steps are
  built, and resolves the flags into a frozen `RunSpec`. `settings.py` reads
  `config/settings.yaml`, else the packaged example, as typed machine values
  plus an optional `defaults:` block keyed by option names; a flag overrides a
  settings default, which overrides the code default. Each output folder holds
  one SQLite store, `autoexcerpter.sqlite`, with the runs and their effective
  specs, one document per item, one transcription unit and one summary unit
  per page, every provider response as received, the citations and the
  rendered files with their hashes. Five views (`page_text`, `page_summaries`,
  `document_citations`, `document_status`, `page_attempts`) form the read
  contract. A run leases each document it works on and resumes it from its
  stored units; `render` writes the files again and `drop` deletes stored
  documents, both without model calls. The batch layer (`batch/`) submits
  pages and summaries to the batch APIs of OpenAI, Anthropic and Google, and
  `batch check` collects, adopts or abandons the submissions. The logic shared
  with ChronoTranscriber and ChronoMiner is vendored byte-identically in
  `common/` and pinned by `MANIFEST.json`: the run store and its SQLite writer
  (`store`, `sqlite`), the provider request builder (`provider_requests`,
  `request_options`), the capability table and its sourced vendor facts
  (`capability_table`, `capability_facts`, `capabilities`), image presets and
  payloads (`image_presets`, `images`, `native`), retry and rate limits
  (`retry`, `rate_limit`, `http_timeouts`), the settings schema and loader
  (`settings_schema`, `settings_loader`), the wizard (`wizard`, `prompts`,
  `options`, `standard_options`), structured calls and attempt capture
  (`structured`, `capture`, `responses`, `usage`), the batch backends (`batch`,
  `batch_openai`, `batch_anthropic`, `batch_google`, `batch_reconcile`,
  `polling`), and the agent contract and heartbeat. AutoExcerpter's own logic
  covers the transcription and summary phases (`llm/`, `pipeline/`), citation
  consolidation and OpenAlex enrichment (`rendering/citations/`), the
  correction of printed page numbers (`rendering/page_numbering.py`), and the
  Markdown and DOCX renderers with native Word equations (`rendering/`). The
  extension seam `ext.py` ships pass-through hooks that a local access layer
  can replace: `register_args`, `extra_argv`, `settings_blocks`,
  `wizard_steps`, `startup`, `batch_allowed`, `key_variables`,
  `describe_access`, `client_options` and the `attempt` context around every
  model and batch call.
