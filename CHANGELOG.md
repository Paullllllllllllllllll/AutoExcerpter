# Changelog

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
