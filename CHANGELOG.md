# Changelog

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
