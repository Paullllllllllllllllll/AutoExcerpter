# AutoExcerpter v1.0.2

AutoExcerpter transcribes PDFs and folders of page images with vision-language
models, summarizes each page and builds a consolidated bibliography whose entries
are enriched from OpenAlex. It runs on OpenAI, Anthropic, Google and OpenRouter
models and on OpenAI-compatible endpoints that you name in the settings. The
`run` command serves scripts and agents; the `ui` command walks you through a run
in the terminal. Please report bugs on the
[GitHub issue tracker](https://github.com/Paullllllllllllllllll/AutoExcerpter/issues).

## Overview

An item is one PDF or one folder of page images (PNG, JPEG, TIFF, BMP, GIF or
WebP). In the transcription stage, every page is rendered or loaded in memory,
sized to the model's image limits and sent to the transcription model, which
returns the page text with Markdown structure, LaTeX math, footnotes, figure
descriptions and the printed page number. AutoExcerpter cleans the text and writes
the pages in scan order. In the summary stage (on by default), the summary model
condenses each content page into bullet points and lists the references it cites.
AutoExcerpter then corrects page numbers across the document, merges duplicate
citations, looks them up in OpenAlex and renders the summary as Markdown and Word
files.

For an input `X.pdf` or an image folder `X/`, a run writes `X_transcription.md`,
`X_summary.md`, `X_summary.docx`, the working logs in `X_autoexcerpter/` and the
folder database `autoexcerpter.sqlite` ([Output Contract](#output-contract)).

## Installation

AutoExcerpter needs Python 3.13 or later and [uv](https://docs.astral.sh/uv/).
Install it from a clone, since the settings files are read from the repository's
`config/` folder.

```bash
git clone https://github.com/Paullllllllllllllllll/AutoExcerpter.git
cd AutoExcerpter
uv sync                      # development: uv sync --all-extras
cp config/settings.example.yaml config/settings.yaml
```

On Windows, copy the file with `copy config\settings.example.yaml
config\settings.yaml`. Then set the API key of each provider you use as an
environment variable. The settings name the variables (`api_keys:`); the defaults
are `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` and
`OPENROUTER_API_KEY`. An OpenAlex key in `OPENALEX_API_KEY` is optional.

```bash
export OPENAI_API_KEY="sk-..."        # PowerShell: $env:OPENAI_API_KEY = "sk-..."
```

## Quick Start

```bash
uv run autoexcerpter run --input paper.pdf           # outputs beside the input
uv run autoexcerpter run --input scans --all --output excerpts
uv run autoexcerpter run --input scans --select 1-3  # or a name: --select braudel
uv run autoexcerpter run --input scans --all --dry-run --json
uv run autoexcerpter ui                              # guided run
```

A folder input is searched recursively; item numbers count in discovery order,
PDFs first. Relative paths resolve against the current directory. With
`uv --directory <clone> run autoexcerpter ...` the current directory is the
clone, so pass absolute paths.

## Command Reference

The tables list every option of `autoexcerpter run --help` with its help wording
and default. They mirror the option table (`autoexcerpter/spec.py`), and
`tests/cli/test_readme_options.py` checks them against it. Options without a
stated default are off or unset until given. The last column says whether the
settings `defaults:` block can set the option; its key there is the flag without
the leading dashes, with underscores for hyphens (`--transcription-format` becomes
`transcription_format`). `autoexcerpter ui` takes only `--settings FILE`.

### Input

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--input PATH` | PDF file, image folder, or folder of PDFs and image folders | | yes |
| `--select EXPR` | select items by number (1,3), range (1-5) or name search | | yes |
| `--all` | process every discovered item | | yes |

`--input` is required. A single discovered item needs neither `--select` nor
`--all`; several items without either are a usage error. A name search matches
part of a file or folder name, ignoring case.

### Output

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--output DIR` | output folder | beside the input | yes |
| `--beside-input` | write outputs beside each input | | no |
| `--transcription-format {md,txt}` | transcription file format | md | yes |
| `--formats LIST` | comma-separated output formats; from summary-md, summary-docx, sqlite | summary-md,summary-docx,sqlite | yes |
| `--keep-working-files`, `--no-keep-working-files` | keep the X_autoexcerpter/ working logs of an item completed under --force; runs without --force always keep them | off | yes |

The transcription file is always written. `--beside-input` overrides an `output`
settings default.

### Summary

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--summarize`, `--no-summarize` | summarize after transcribing | on | yes |
| `--context TEXT\|PATH\|auto\|none` | summary context: topics as text or a path to a text file (overrides sidecars), auto (the item's file or folder sidecar) or none (no context) | auto | yes |
| `--fallback-context TEXT\|PATH\|none` | summary context for items without a sidecar under --context auto: topics as text, a path to a text file, or none | none | yes |

Sidecars are text files with one topic per line: `X_summary_context.txt` beside
the input, else `F_summary_context.txt` beside the folder `F` that holds the
input. A value that names an existing file is read as a file, any other value as
context text.

### Citations

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--openalex`, `--no-openalex` | enrich citations with OpenAlex metadata | on | yes |

### Model

The unprefixed flag sets both phases; the `--transcription-*` and `--summary-*`
flags set one phase and win over it. The `defaults:` block can set every model
option, for example as `model` or `summary_verbosity`.

| Setting | Values | Both phases | Transcription | Summary |
| --- | --- | --- | --- | --- |
| model provider | `{openai,anthropic,google,openrouter,custom}` | `--provider`: inferred from the model name | `--transcription-provider`: inferred from the model name | `--summary-provider`: inferred from the model name |
| model name; a provider: prefix (anthropic:NAME) selects the provider | `NAME` | `--model`: per phase | `--transcription-model`: gpt-5.6-luna | `--summary-model`: gpt-5.6-luna |
| named custom endpoint from the settings (implies provider custom) | `NAME` | `--endpoint`: none | `--transcription-endpoint`: none | `--summary-endpoint`: none |
| reasoning effort | `{none,minimal,low,medium,high,xhigh,max}` | `--reasoning-effort`: per phase | `--transcription-reasoning-effort`: high | `--summary-reasoning-effort`: high |
| output verbosity | `{low,medium,high}` | `--verbosity`: per phase | `--transcription-verbosity`: medium | `--summary-verbosity`: low |
| maximum output tokens | `N` | `--max-output-tokens`: the model's output limit | `--transcription-max-output-tokens`: the model's output limit | `--summary-max-output-tokens`: the model's output limit |
| sampling temperature, 0.0 to 2.0 | `T` | `--temperature`: per phase | `--transcription-temperature`: 1.0 | `--summary-temperature`: 1.0 |
| nucleus sampling, 0.0 to 1.0; sent only when given and supported | `P` | `--top-p`: not sent | `--transcription-top-p`: not sent | `--summary-top-p`: not sent |
| OpenAI service tier; only for provider openai | `{auto,default,flex,priority}` | `--service-tier`: per phase | `--transcription-service-tier`: flex | `--summary-service-tier`: flex |
| schema (API-enforced), json (prompted, validated) or text (plain) | `{schema,json,text}` | `--response-format`: schema when the model supports it, else json | `--transcription-response-format`: schema when the model supports it, else json | `--summary-response-format`: schema when the model supports it, else json |

### Images

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--dpi native\|N` | render resolution: native or a DPI value | the provider's recommended value | yes |
| `--image-detail {low,high,auto,original}` | image detail sent to the model | the provider's recommended value | yes |
| `--image-format {jpeg,png}` | page image format | the provider's recommended value | yes |
| `--jpeg-quality N` | JPEG quality, 1 to 100 | the provider's recommended value | yes |
| `--grayscale`, `--no-grayscale` | convert page images to grayscale | the provider's recommended value | yes |

### Run Control

| Flag | Description | Default | In `defaults:` |
| --- | --- | --- | --- |
| `--force` | reprocess items whose outputs exist (resume is the default) | | no |
| `--retranscribe` | transcribe again instead of reusing logged transcriptions | | no |
| `--concurrency N` | parallel page requests | 16 | yes |
| `--dry-run` | plan without API calls or writes | | no |
| `--json` | print one JSON summary line on stdout | | no |
| `--settings FILE` | alternate settings file | config/settings.yaml | no |

### Exit Codes and JSON Summary

| Code | Meaning |
| --- | --- |
| 0 | every item completed, or nothing was left to do |
| 1 | an item failed, or no item could be processed (no PDF or image folder found, `--select` matched nothing, an unexpected error) |
| 2 | usage or configuration error (invalid flags or settings, several items without `--all` or `--select`, two items with one output name in one folder, a working log in the output folder that belongs to another input of the same name, summaries on without a summary format, an unset API key variable) |
| 130 | interrupted (Ctrl+C) |

Progress, warnings and errors go to stderr. With `--json`, stdout carries exactly
one JSON line on every exit, errors and interrupts included. Each line holds
`dry_run`, which is true whenever `--dry-run` was given. A run summary adds:

| Field | Content |
| --- | --- |
| `items_total` | items processed in this run |
| `items_complete` | items that finished with every page and output |
| `items_failed` | items that finished incomplete |
| `items_skipped` | items skipped because they were already complete |
| `outputs` | absolute paths of the files written, item by item, then the databases |
| `usage` | tokens per role (`transcription`, `summary`) and their sum under `total`, each with `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens` (input read from the provider's prompt cache) and `reasoning_tokens` |

A dry run that plans successfully prints the plan instead: `to_process` lists each
item's `name`, resume `state` and `completed_pages`; `skipped` lists the complete
items; `spec` maps every field of the effective run specification, such as
`transcription_model.model`, to its `value` and `source` (`flag`, `settings` or
`default`). The plan has no `usage` field, since a dry run makes no calls.

## Guided UI

`autoexcerpter ui` asks one step at a time: input (with the PDFs, image folders
and pages found, and the item selection), output location, summary on or off,
formats, summary context, transcription model, summary model, page images
("Recommended" or "Customize") and run options. Every list ends with Back; a text
or path prompt takes `<` alone as Back. Settings defaults are preselected, and
answers act like flags over them. The review screen marks values taken from
settings defaults with `(s)`, previews the plan and prints the equivalent command:

```text
Review                                            (s) = from settings defaults
  input                 /data/docs/doc.pdf  (1 PDF, 3 pages)
  transcription model   openai / gpt-5.6-terra (s) | reasoning high | tier priority (s)
  run                   resume | OpenAlex on | concurrency 16

Plan: 1 new, 0 partial, 0 complete
  doc: new

Equivalent command:
  autoexcerpter run --input /data/docs/doc.pdf --model gpt-5.6-terra
    --service-tier priority
? Start:
 > Run
   Edit a step
   Copy command and quit
   Quit
   Back
```

The command spells out every value that differs from the code default, settings
defaults included, so it reproduces the run on another machine. On Windows it is
quoted for `cmd`; in PowerShell a `$` inside a double-quoted value is expanded,
so check such values before pasting. Run shows progress bars and ends with a
summary table. Edit a step returns to one step. Copy command and quit prints the
command on stdout and copies it to the clipboard where one is available. Quit
exits. When items are already complete, the review offers to reprocess them
(`--force`). Run is withheld while the plan fails or a needed API key variable is
unset; the review names the variable. The UI needs an interactive terminal on
stdin and stdout; anything else exits with code 2.

## Settings File

AutoExcerpter reads `config/settings.yaml` (gitignored). When it is missing, the
tool reads the tracked `config/settings.example.yaml` and logs one informational
line that asks you to copy it. `--settings FILE` reads another file, which must
exist. Every key is optional and keeps its code value when missing; unknown keys
are an error that names the file and the key. The example file, with its
`defaults:` block shown uncommented:

```yaml
api_keys: {openai: OPENAI_API_KEY, anthropic: ANTHROPIC_API_KEY, google: GOOGLE_API_KEY, openrouter: OPENROUTER_API_KEY}
endpoints: {}                       # named endpoints, see Providers
timeouts: {request: 900, connect: 10, write: 30, pool: 30}   # seconds
rate_limits: [[10, 1], [600, 60], [600, 3600]]   # [max_requests, window_seconds]
retry:
  max_attempts: 8                   # every call, the first included
  backoff_base: 0.5
  backoff_cap: 120
  backoff_multipliers: {rate_limit: 2.0, timeout: 1.5, connection: 1.5, server_error: 2.0, other: 2.0, validation: 1.5}
  jitter: {min: 0.5, max: 1.0}
  schema_retries:
    transcription:
      validation_failure: {enabled: true, max_attempts: 3, backoff_base: 0.5, backoff_multiplier: 1.5}
      no_transcribable_text: {enabled: false, max_attempts: 0, backoff_base: 0.1, backoff_multiplier: 1.5}
      transcription_not_possible: {enabled: true, max_attempts: 3, backoff_base: 0.1, backoff_multiplier: 1.5}
    summary:
      validation_failure: {enabled: true, max_attempts: 3, backoff_base: 0.5, backoff_multiplier: 1.5}
state_dir: ''                       # '' means ~/.autoexcerpter
openalex: {email: '', api_key_env: OPENALEX_API_KEY, max_requests: 300}
defaults:
  model: gpt-5.6-terra
  reasoning_effort: high
  summary_verbosity: low
  service_tier: flex
  transcription_format: md
  formats: [summary-md, summary-docx, sqlite]
  concurrency: 16
  fallback_context: path/to/summary_topics.txt
```

`api_keys` names the variable that holds each provider's key, never the key.
`timeouts.request` is the read timeout; 900 s suits the OpenAI flex tier, which
may queue requests. Anthropic and Google take `request` as their only timeout.
The rate limits hold per provider. A failed API call waits `backoff_base *
multiplier^n` plus jitter, at most `backoff_cap` seconds, a server's
`Retry-After` included. `schema_retries` covers retries driven by the model
output: invalid JSON (`validation_failure`) or a flag the model sets on a page.
`state_dir` and `openalex` are described under
[Citations and OpenAlex](#citations-and-openalex).

The `defaults:` block holds run defaults keyed by option names. Unknown keys,
options that only the command line can set and invalid values are errors.
Precedence is flag (or wizard answer), then settings default, then code default.
`--dry-run --json` prints the effective spec with the source of each value, and
the `documents` table of the database stores it for every item, so a run stays
reproducible after the defaults change. Tuning values such as the citation merge
thresholds, image byte caps and text cleaning switches are code constants.

## Providers and Response Formats

Each phase's provider comes from `--provider` (or a phase flag), else a
`provider:` prefix on the model name (`anthropic:claude-sonnet-5`), else a
`vendor/model` name (OpenRouter), else the name itself (`claude` is Anthropic,
`gemini` is Google, `gpt`, `chatgpt`, `o1`, `o3`, `o4` and `text-` are OpenAI),
else OpenAI. A prefix that contradicts a provider given at the same or a higher
precedence level is an error. The key comes from the variable `api_keys:` names.
Custom endpoints are named in the settings and selected with `--endpoint NAME`,
which implies provider `custom`:

```yaml
endpoints:
  local:
    base_url: https://your-endpoint.example.com/v1/
    api_key_env: YOUR_ENDPOINT_API_KEY
    supports_vision: true    # the model accepts images (default true)
    supports_schema: false   # the API enforces JSON schemas (default false)
```

`--response-format schema` has the API enforce the JSON schema, `json` asks for
JSON in the prompt and validates it with retries, and `text` asks for plain text.
Unset, a phase uses `schema` when the model supports structured output and `json`
otherwise; a custom endpoint uses `schema` when it declares `supports_schema`. A
requested `schema` on a model without structured output falls back to `json`
with a warning. The summary always asks for the summary schema; with `text` it
keeps a plain answer that fails validation as one bullet point.

Reasoning effort reaches OpenAI reasoning models as `reasoning.effort`. Claude
models with adaptive thinking receive an effort level, other Claude reasoning
models a thinking limit of 2,048 (`low`) to 16,384 (`xhigh`) tokens; Gemini 3
models receive a thinking level, earlier Gemini models a limit of 512 to 16,384
tokens. OpenRouter and custom endpoints receive no reasoning parameter. Verbosity
reaches the OpenAI models that support it. Temperature and top-p are left out
while reasoning is active and for models that reject them. Without
`--max-output-tokens`, OpenAI, Anthropic and Google models get their family's
output limit. The service tier is sent with OpenAI calls only; a tier given for
another provider by flag or settings default is ignored with a warning on stderr.

The family rules in `autoexcerpter/llm/capabilities.py` feed the shared resolver
in `autoexcerpter/common/capabilities.py`. Rules match the lowercased model name
in order and the first match wins, so specific names (`gpt-5.6-terra`) precede
general ones (`gpt-5`). A rule sets image input, structured output, reasoning,
verbosity, sampling parameters, context and output limits and image size caps.
Any model name is accepted. One that no rule matches gets a conservative profile
(no image input, no structured output, no reasoning parameters) and a warning.

The image options default to the preset of the transcription model's family.
OpenRouter models use the family their name points to (Claude, Gemini, else
OpenAI); custom endpoints use the custom preset.

| Family | `--dpi` | `--image-detail` | Format | JPEG quality | Grayscale | Base64 cap |
| --- | --- | --- | --- | --- | --- | --- |
| openai | native | original | jpeg | 95 | yes | 50 MB |
| anthropic | native | auto | jpeg | 95 | yes | 10 MB |
| google | 300 | high | jpeg | 95 | yes | 20 MB |
| custom | 150 | high | jpeg | 85 | yes | none |

`native` renders each PDF page at the density of its scan; pages without a
page-spanning scan render at 300 DPI. Payloads are sized to the model's documented
limits. `--image-detail` is the request detail for OpenAI models and selects the
local resize profile for the others (the media resolution for Google). An
oversized PNG falls back to JPEG with a warning; a JPEG over the cap fails the
page.

## Output Contract

For an input `X.pdf` or an image folder `X/`, written beside the input or into
`--output`:

```text
X_transcription.md       (X_transcription.txt with --transcription-format txt)
X_summary.md             summary-md
X_summary.docx           summary-docx
X_autoexcerpter/         transcription.jsonl, summary.jsonl (working logs)
autoexcerpter.sqlite     sqlite; one database per output folder
```

The transcription starts with a metadata header and holds the pages in scan
order. A printed page number follows its page as `<page_number>12</page_number>`.
A page without one is preceded by `<page_break pdf="N"/>` (`image="N"` for image
folders), N being its 1-based position. A page-number tag glued to a word is a
note marker and becomes a footnote reference (`things.[^10]`).

The summary files list the document structure, one section per content page
headed by its printed page number (or `[No printed number; PDF p. N]`, with an
inferred number in brackets, `Page [6]`) and the consolidated references with
their pages. Page numbers are checked per section (front matter, body,
appendix) against runs of consecutive detected numbers
(`pipeline/page_numbering.py`): a misread number inside a run is corrected, while
an excerpt that skips pages, say from 12 to 30, keeps both runs. The Markdown file
keeps LaTeX math as `$...$` and `$$...$$`; the
Word file converts it to native Word equations (`rendering/equations.py`) and
links references matched in OpenAlex.

The working logs are versioned JSONL: a header with the format version, the model
and the image settings fingerprint, then one line per page.
`autoexcerpter.sqlite` (`rendering/sqlite.py`) has four tables. `documents` holds
one row per item: input path and content hash, tool version, models, settings
fingerprint, the effective spec with sources, page counts, completeness and the
files written. `pages` holds each page's status, error, model, token usage and
text; `summaries` each page summary with bullet points and references; and
`citations` the consolidated citations with pages, DOI, URL and OpenAlex
metadata. An item's rows are upserted in one transaction, so resumed and repeated
runs update them in place, and two processes can share a database.

When an item's longest path would exceed the Windows path limit,
`pipeline/paths.py` shortens the item name and appends an 8-character hash of the
full name, the same way for every file of the item. `X_autoexcerpter/` stays
after a run unless the item completed under `--force` without
`--keep-working-files`.

## Batch Processing

AutoExcerpter has no provider batch mode; every page is a direct request. A folder
run processes the selected items one after another, with up to `--concurrency`
pages of an item (default 16) in flight at once. The rate limits apply per
provider across the run. A failed item is counted, and the run continues.

## Resume

Resume is the default. Each item is classified from its outputs and working logs
(`pipeline/resume.py`) before the run; a dry run shows the states.

| State | Meaning | Action |
| --- | --- | --- |
| `none` | nothing to reuse | process from scratch |
| `partial` | the transcription log holds completed pages | transcribe the remaining pages |
| `transcription_only` | the transcription exists, but summary outputs are missing or the logs show missing or failed pages | redo what is missing or failed |
| `complete` | every expected output exists and the logs show no gap or failure | skip (`items_skipped`) |

With the `sqlite` format, a complete item without a complete `documents` row is
finished from its logs without model calls; without logs it stays skipped with a
warning that `--force` rebuilds it. A logged failure never counts as complete, so
a rerun retries failed pages. A page stopped by a content filter or refused by
the model fails after one call (`error_type` `content_filter` or `refusal`) and is
retried the same way. An input whose size or image set changed since its log was
written is processed from scratch. When an item's working log was written for
another input of the same name that still exists, as with two `doc.pdf` files
sharing one `--output` folder, the plan stops with exit 2 before any work; give one
of them another output folder.

`--force` processes every selected item from scratch. `--retranscribe` keeps
resume but transcribes logged pages again instead of reusing them. Before logged
pages are reused while pages remain to transcribe, the run's image settings must
match those recorded in the log. Otherwise the item fails with the changed keys
named and a hint to use `--force`; a log without a fingerprint is refused the same
way. Switching the transcription model on resume is allowed: the model and the
settings derived from it (detail resolution, image cap) are not compared, the run
warns, and the transcription file notes both models. Each page record and its
`pages.model` row name the model that transcribed the page. A log of another
format version is ignored with a warning, so its item starts over.

## Citations and OpenAlex

Citations from all pages are merged by a normalized key, then near-duplicates with
the same first author and year are merged conservatively; different years or
volumes never merge. An in-text stub (author and year only) joins its full
reference when exactly one matches. Pages are consolidated per numbering system
and never mixed in one range: `pp. 5-7; PDF p. 12`, `pp. x-xii`, `Image 7`, and
`p. [6]` for an inferred number.

Each unique citation without a cached result is looked up by DOI when the text
holds one, else by title filtered to the cited year plus or minus one, else by a
free-text search. A candidate links only when its title overlaps the citation and
its year or an author surname confirms the match. A lookup ends in one of four
outcomes:

| Outcome | Cached in `state_dir` | Effect |
| --- | --- | --- |
| found | `openalex_cache.json` | metadata, DOI and link attached |
| not found | `openalex_misses.json` | not looked up again |
| transient failure | no | looked up again in a later run |
| budget exhausted | no | a 429 with a long `retryAfter`; no further requests in this run |

The cache is written every 25 lookups and when enrichment ends, an interruption
included; cached results are served even when the request budget is exhausted.
The `openalex:` settings hold the contact `email`, the variable with an optional
API key (`api_key_env`; the key is redacted from logs) and `max_requests`, the
citations looked up per document. `--no-openalex` turns lookups off.

## Architecture

```text
autoexcerpter/
  __main__.py   entry point: dispatch run or ui
  spec.py       RunSpec dataclasses and the option table (OPTIONS)
  settings.py   typed settings: machine values and the defaults block
  run.py        core: discover, plan, run
  events.py     run events: item started, page done, waiting, warning, item finished
  ext.py        extension seam with pass-through defaults
  providers.py  provider names and inference from model names
  cli/          argparse from the option table, stderr reporter, --json, exit codes
  ui/           questionary wizard, review screen, rich progress
  common/       vendored modules shared with the sibling tools
  llm/          client factory, model calls, capability table, prompts
  imaging/      page payload sources and image presets
  pipeline/     item processing, pages, working logs, resume, context, paths
  rendering/    transcription, Markdown, DOCX, equations, SQLite, citations/
  resources/    prompts and the transcription and summary schemas
```

The core imports nothing from `cli/` or `ui/`. It never prints, prompts or exits
and reports through the run events, which the CLI writes to stderr and the UI
shows as progress bars. `common/` holds tool-agnostic modules (option tables,
wizard framework, settings loader, structured calls, retry, rate limiting, HTTP
timeouts, images, JSONL, SQLite writer, agent contract) kept byte-identical
across the sibling tools: `common/MANIFEST.json` pins each file by its SHA-256, and
`tests/common/test_manifest.py` fails when a copy drifts. `ext.py` holds the
extension hooks (extra arguments, settings blocks and wizard steps, a startup
hook, an attempt context around every model call); its defaults pass everything
through.

## Development

```bash
uv sync --all-extras
uv run pytest                      # hermetic suite
uv run ruff check .
uv run ruff format --check .
uv run mypy autoexcerpter tests    # strict
markdownlint-cli2 README.md
```

The suite removes provider keys from the environment, points the home folder at
a temporary directory, blocks non-loopback connections and fails any test that
writes outside its temporary directory. Model calls go to a scripted fake chat
model behind the real client. Golden files hold the help texts, a recorded wizard
session, the outputs per provider and response format, and DOCX, Markdown and
SQLite read-backs. After an intended change, rewrite them with
`AE_UPDATE_GOLDEN=1` (PowerShell: `$env:AE_UPDATE_GOLDEN = "1"`) and review the
diff.

## Troubleshooting

**API key not found.** A run whose model key variable is unset stops before any
work, names the variable and exits with code 2; the UI withholds Run and names
the variable on the review screen.

**Image settings changed.** A resumed item with pages left to transcribe fails
when its image settings differ from its log. Rerun with the earlier
settings, or pass `--force` to start the item over.

**Pages failed with `content_filter` or `refusal`.** A provider's content filter
can stop pages dense with long verbatim passages. Rerun the item with another
model; since the model is part of the image settings, this needs `--force`.

**Rate limit errors (429) or timeouts.** Lower `--concurrency`, adjust
`rate_limits` to your provider account, or raise `timeouts.request`.

**"No exact capability profile" warning.** The model name matches no rule, so the
conservative profile applies. A model on an OpenAI-compatible server can run as a
named endpoint whose settings declare its image and schema support.

**No sign of the example settings.** The console shows warnings and errors only,
so a run on `config/settings.example.yaml` does not mention it. Copy the example
to `config/settings.yaml`.

## Versioning

This project follows semantic versioning (`MAJOR.MINOR.PATCH`). The version in
`pyproject.toml` is the single source of truth; it is mirrored in the title
heading above and tagged in git as `vX.Y.Z`. The repository history begins with a
single baseline commit at v1.0.0. Every release is described in
[CHANGELOG.md](CHANGELOG.md).
